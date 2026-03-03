
def power_distance(x):
    """
    Distance function: y = x^(3/4) + (x/50)^(3/2)

    Args:
        x: input value, can be a float, int, or numpy array / torch tensor
    Returns:
        y: computed distance
    """
    return x ** (3 / 4) + (x / 50) ** (3 / 2)
