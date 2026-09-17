"""SparkFun-compatible OLED primitive tests."""
from ssd1306 import OLED, WIDTH, HEIGHT


def _lit_pixels(d: OLED) -> set:
    return {(x, y) for y in range(HEIGHT) for x in range(WIDTH) if d.get(x, y)}


class TestRectangleSideWalls:
    def test_short_box_is_open_ended(self):
        d = OLED()
        d.rectangle(10, 10, 6, 3)
        assert all(d.get(x, 10) for x in range(10, 16))
        assert all(d.get(x, 12) for x in range(10, 16))
        assert all(d.get(x, 11) == 0 for x in range(10, 16))

    def test_tall_box_is_closed(self):
        d = OLED()
        d.rectangle(10, 20, 6, 4)
        assert d.get(10, 21) == 1
        assert d.get(15, 21) == 1
        assert all(d.get(x, 21) == 0 for x in range(11, 15))


class TestRectangleDegenerate:
    def test_width_1_rectangle_equals_a_line(self):
        r = OLED()
        r.rectangle(5, 5, 1, 5)
        line = OLED()
        line.line(5, 5, 5, 9)
        assert _lit_pixels(r) == _lit_pixels(line)
        assert _lit_pixels(r)


class TestLineBresenham:
    def test_non_45_slope_pixel_set_is_pinned(self):
        d = OLED()
        d.line(0, 0, 4, 2)
        assert _lit_pixels(d) == {(0, 0), (1, 0), (2, 1), (3, 1), (4, 2)}


class TestPixelClipping:
    def test_out_of_range_pixel_is_a_noop(self):
        d = OLED()
        d.pixel(WIDTH + 5, 5)
        d.pixel(5, HEIGHT + 5)
        d.pixel(-1, 5)
        d.pixel(5, -1)
        assert _lit_pixels(d) == set()
