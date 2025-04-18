import numpy as np

def spatial_sincos_encoding(lat, lon, dim=16):
    """
    Encodes latitude and longitude into sinusoidal positional encodings.

    Args:
        lat (float): Latitude value in degrees.
        lon (float): Longitude value in degrees.
        d_model (int): Dimension of the positional encoding.

    Returns:
        numpy.ndarray: Sinusoidal positional encoding of shape (2, d_model).
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    encoding = np.zeros((2, dim))
    
    for i in range(0, dim, 2):
        encoding[0, i] = np.sin(lat_rad / (10000 ** (i / dim)))
        encoding[0, i + 1] = np.cos(lat_rad / (10000 ** (i / dim)))
        encoding[1, i] = np.sin(lon_rad / (10000 ** (i / dim)))
        encoding[1, i + 1] = np.cos(lon_rad / (10000 ** (i / dim)))
        
    return encoding



lat = 40.7128  # Latitude of New York City
lon = -74.0060  # Longitude of New York City
encoding = spatial_sincos_encoding(lat, lon, dim=16)
print(encoding)

