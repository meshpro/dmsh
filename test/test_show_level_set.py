import dmsh


def test_show():
    """
    Draw a rectangle of a rectangle.

    Args:
    """
    # geo = dmsh.Circle([0.0, 0.0], 1.0)
    geo = dmsh.Rectangle(-1.0, +1.0, -1.0, +1.0)
    geo.show()


if __name__ == "__main__":
    test_show()
