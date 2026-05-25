from public_detect.review import crop_bounds


def test_crop_bounds_clamps_to_image() -> None:
    assert crop_bounds((5, 5, 15, 15), image_width=20, image_height=20, margin=1.0) == (
        0,
        0,
        20,
        20,
    )


def test_crop_bounds_expands_box() -> None:
    assert crop_bounds((40, 40, 60, 60), image_width=100, image_height=100, margin=0.5) == (
        30,
        30,
        70,
        70,
    )
