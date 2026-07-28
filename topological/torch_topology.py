"""GPU-accelerated topology primitives using torch.fft.

Replaces the numpy FFT bottleneck in decomposition/intervention pipeline.
All functions operate on torch tensors and stay on GPU.
"""
import torch
import torch.fft

TAU = 2.0 * torch.pi


def _poisson_stream_torch(charge: torch.Tensor) -> tuple[torch.Tensor, float]:
    """FFT-based Poisson solver for ∇²ψ = 2πQ. Batched over channels.

    Args:
        charge: (C, H, W) int tensor or (H, W) int tensor
    Returns:
        stream: (C, H, W) float tensor
        residual: float max error
    """
    if charge.ndim == 2:
        charge = charge.unsqueeze(0)
    C, H, W = charge.shape
    q = charge.to(torch.float64)
    device = q.device

    freq_x = TAU * torch.fft.fftfreq(H, device=device)
    freq_y = TAU * torch.fft.fftfreq(W, device=device)
    eigenvalues = (
        2.0 * torch.cos(freq_x)[:, None]
        + 2.0 * torch.cos(freq_y)[None, :]
        - 4.0
    )  # (H, W)

    right_hat = torch.fft.fft2(TAU * q)  # (C, H, W) complex
    stream_hat = torch.zeros_like(right_hat)
    mask = eigenvalues.abs() > 1e-15
    stream_hat[:, mask] = right_hat[:, mask] / eigenvalues[mask]
    stream_hat[:, 0, 0] = 0.0
    stream = torch.fft.ifft2(stream_hat).real  # (C, H, W)

    # Residual check
    laplacian = (
        torch.roll(stream, -1, dims=1)
        + torch.roll(stream, 1, dims=1)
        + torch.roll(stream, -1, dims=2)
        + torch.roll(stream, 1, dims=2)
        - 4.0 * stream
    )
    residual = float((laplacian - TAU * q).abs().max().cpu())

    if C == 1:
        stream = stream.squeeze(0)
    return stream, residual


def canonical_vortex_field_torch(charge: torch.Tensor) -> torch.Tensor:
    """Construct canonical unit vortex field on GPU. Batched over channels.

    Args:
        charge: (C, H, W) int tensor
    Returns:
        field: (C, H, W) complex128 tensor, unit magnitude
    """
    if charge.ndim == 2:
        charge = charge.unsqueeze(0)
    C, H, W = charge.shape
    device = charge.device

    stream, _ = _poisson_stream_torch(charge)  # (C, H, W)
    if stream.ndim == 2:
        stream = stream.unsqueeze(0)

    # Backward-difference links (matching numpy canonical_vortex_field convention)
    link_y = torch.roll(stream, 1, dims=2) - stream  # (C, H, W)
    link_x = stream - torch.roll(stream, 1, dims=1)   # (C, H, W)

    # Remove global holonomy
    holonomy_x = torch.angle(torch.exp(1j * link_x.sum(dim=1)))  # (C, H)
    holonomy_y = torch.angle(torch.exp(1j * link_y.sum(dim=2)))  # (C, W)
    link_y = link_y - holonomy_y[:, None, :] / H
    link_x = link_x - holonomy_x[:, :, None] / W

    # Integrate links to field
    phasor_x = torch.exp(1j * link_y)  # (C, H, W) — note swapped axes
    phasor_y = torch.exp(1j * link_x)
    field = torch.ones(C, H, W, dtype=torch.complex128, device=device)

    # Integrate along x
    for i in range(H - 1):
        field[:, i + 1, 0] = phasor_x[:, i, 0] * field[:, i, 0]
    # Integrate along y
    for j in range(W - 1):
        field[:, :, j + 1] = phasor_y[:, :, j] * field[:, :, j]

    field = field / field.abs()

    if C == 1:
        field = field.squeeze(0)
    return field


def extract_charge_torch(field: torch.Tensor, tolerance: float = 1e-5) -> torch.Tensor:
    """Extract integer charge map from complex field. Batched over channels.

    Args:
        field: (C, H, W) complex tensor
    Returns:
        charge: (C, H, W) int tensor
    """
    if field.ndim == 2:
        field = field.unsqueeze(0)
    C, H, W = field.shape
    device = field.device

    # Forward-difference links
    unit = field / (field.abs() + 1e-12)
    link_x = torch.angle(torch.roll(unit, -1, dims=1) * unit.conj())
    link_y = torch.angle(torch.roll(unit, -1, dims=2) * unit.conj())

    # Plaquette curl
    curl = link_x + torch.roll(link_y, -1, dims=1) - torch.roll(link_x, -1, dims=2) - link_y
    charge = torch.round(curl / TAU).to(torch.int64)

    residual = float((curl - TAU * charge.float()).abs().max().cpu())
    if residual > tolerance:
        raise RuntimeError(f"integer-charge residual exceeds tolerance: {residual}")

    if C == 1:
        charge = charge.squeeze(0)
    return charge


def decompose_torch(field: torch.Tensor) -> dict:
    """Decompose complex field into magnitude, vortex, smooth on GPU.

    Args:
        field: (C, H, W) complex tensor
    Returns:
        dict with 'magnitude', 'vortex', 'smooth', 'charge' tensors
    """
    if field.ndim == 2:
        field = field.unsqueeze(0)
    C, H, W = field.shape
    device = field.device

    magnitude = field.abs()
    charge = extract_charge_torch(field)
    canonical = canonical_vortex_field_torch(charge)  # (C, H, W)
    compact = field / (magnitude + 1e-12)
    smooth = compact * canonical.conj()

    if C == 1:
        magnitude = magnitude.squeeze(0)
        canonical = canonical.squeeze(0)
        smooth = smooth.squeeze(0)

    return {
        "magnitude": magnitude,
        "vortex": canonical,
        "smooth": smooth,
        "charge": charge if C > 1 else charge.squeeze(0),
    }


def decompose_from_hidden(hidden: torch.Tensor) -> dict:
    """Decompose hidden state (B, 2, C, H, W) into components on GPU.

    This is the GPU equivalent of decompose_hidden().
    Processes all batch items and channels in one batched call.

    Args:
        hidden: (B, 2, C, H, W) float tensor (real+imag stacked)
    Returns:
        dict with batched 'magnitude', 'vortex', 'smooth', 'charge' tensors
    """
    B, _, C, H, W = hidden.shape
    # Convert to complex: (B, C, H, W)
    field = torch.complex(hidden[:, 0], hidden[:, 1])  # (B, C, H, W)
    # Flatten batch+channel for batched FFT
    field_flat = field.reshape(B * C, H, W)  # (B*C, H, W)
    result = decompose_torch(field_flat)
    # Reshape back
    return {
        "magnitude": result["magnitude"].reshape(B, C, H, W),
        "vortex": result["vortex"].reshape(B, C, H, W),
        "smooth": result["smooth"].reshape(B, C, H, W),
        "charge": result["charge"].reshape(B, C, H, W),
    }
