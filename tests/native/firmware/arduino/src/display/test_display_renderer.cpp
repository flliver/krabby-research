#include <stdint.h>
#include <string.h>

#include "src/display/display_frame_model.h"
#include "src/display/display_renderer.h"

#include "unity.h"

namespace
{

enum CallKind
{
    CALL_FONT,
    CALL_ERASE,
    CALL_PIXEL,
    CALL_LINE,
    CALL_RECT,
    CALL_FILL,
    CALL_TEXT,
};

struct Call
{
    CallKind kind;
    int x0, y0, x1, y1;
    int color;
    char text[24];
};

// Large enough to record a full frame without truncation.
const int CALL_CAPACITY = 4096;

class RecordingCanvas
{
public:
    RecordingCanvas() : count_(0), hasOverflowed_(false) {}

    void useStatusFont() { push(CALL_FONT, 0, 0, 0, 0, 0, ""); }
    void erase() { push(CALL_ERASE, 0, 0, 0, 0, 0, ""); }
    void pixel(int x, int y) { push(CALL_PIXEL, x, y, 0, 0, 0, ""); }

    void line(int x0, int y0, int x1, int y1)
    {
        push(CALL_LINE, x0, y0, x1, y1, 0, "");
    }

    void rectangle(int x, int y, int width, int height)
    {
        push(CALL_RECT, x, y, width, height, 0, "");
    }

    void rectangleFill(int x, int y, int width, int height, int color)
    {
        push(CALL_FILL, x, y, width, height, color, "");
    }

    void text(int x, int y, const char *value)
    {
        push(CALL_TEXT, x, y, 0, 0, 0, value);
    }

    void reset()
    {
        count_ = 0;
        hasOverflowed_ = false;
    }

    int count() const { return count_; }
    bool hasOverflowed() const { return hasOverflowed_; }
    const Call &at(int index) const { return calls_[index]; }

    int countOf(CallKind kind) const
    {
        int found = 0;
        for (int i = 0; i < count_; ++i)
            if (calls_[i].kind == kind)
                ++found;
        return found;
    }

private:
    static int narrow(int value) { return static_cast<uint8_t>(value); }

    void push(CallKind kind, int x0, int y0, int x1, int y1, int color,
              const char *value)
    {
        if (count_ >= CALL_CAPACITY)
        {
            hasOverflowed_ = true;
            return;
        }
        Call &call = calls_[count_++];
        call.kind = kind;
        call.x0 = narrow(x0);
        call.y0 = narrow(y0);
        call.x1 = narrow(x1);
        call.y1 = narrow(y1);
        call.color = color;
        strncpy(call.text, value, sizeof(call.text) - 1);
        call.text[sizeof(call.text) - 1] = '\0';
    }

    Call calls_[CALL_CAPACITY];
    int count_;
    bool hasOverflowed_;
};

DisplayFrame baseFrame()
{
    DisplayFrame frame;
    frame.role = ROLE_FRONT;
    for (BoardRole role : ALL_BOARD_ROLES)
        frame.controllers[role] = true;
    for (ActuatorId actuatorId = ActuatorId::FLHY;
         actuatorId < ActuatorId::ActuatorCount;
         ++actuatorId)
        frame.actuators[actuatorId] = ActuatorGlyph::Hold;
    frame.packVoltage = Volts(26.5f);
    frame.batteryLevel[0] = 0.8f;
    frame.batteryLevel[1] = 0.6f;
    return frame;
}

void setDisplayActuatorGlyph(
    DisplayFrame &frame,
    ActuatorId actuatorId,
    ActuatorGlyph glyph)
{
    frame.actuators[actuatorId] = glyph;
}

void renderThenCapture(
    RecordingCanvas &canvas,
    const DisplayFrame &first,
    const DisplayFrame &second)
{
    DisplayRenderer<RecordingCanvas> renderer(canvas);
    renderer.render(first);
    canvas.reset();
    renderer.render(second);
    TEST_ASSERT_FALSE(canvas.hasOverflowed());
}

}  // namespace

void setUp() {}
void tearDown() {}

void test_the_first_frame_erases_and_rules_the_header(void)
{
    RecordingCanvas canvas;
    DisplayRenderer<RecordingCanvas> renderer(canvas);

    TEST_ASSERT_TRUE(renderer.render(baseFrame()));
    TEST_ASSERT_FALSE(canvas.hasOverflowed());
    TEST_ASSERT_EQUAL_INT(1, canvas.countOf(CALL_ERASE));

    bool hasHeaderRule = false;
    for (int i = 0; i < canvas.count(); ++i)
    {
        const Call &call = canvas.at(i);
        if (call.kind == CALL_LINE && call.x0 == 0 && call.x1 == 127 &&
            call.y0 == SSD1306_HEADER_RULE_Y && call.y1 == SSD1306_HEADER_RULE_Y)
            hasHeaderRule = true;
    }
    TEST_ASSERT_TRUE(hasHeaderRule);
}

void test_an_unchanged_model_draws_nothing(void)
{
    RecordingCanvas canvas;
    DisplayRenderer<RecordingCanvas> renderer(canvas);
    const DisplayFrame frame = baseFrame();

    TEST_ASSERT_TRUE(renderer.render(frame));
    canvas.reset();

    TEST_ASSERT_FALSE(renderer.render(frame));
    TEST_ASSERT_EQUAL_INT(0, canvas.count());
}

void test_a_tilt_change_redraws_only_the_tilt_field(void)
{
    RecordingCanvas canvas;
    DisplayFrame tilted = baseFrame();
    tilted.roll = Degrees(7.0f);

    renderThenCapture(canvas, baseFrame(), tilted);

    TEST_ASSERT_EQUAL_INT(0, canvas.countOf(CALL_ERASE));
    TEST_ASSERT_EQUAL_INT(1, canvas.countOf(CALL_TEXT));
    for (int i = 0; i < canvas.count(); ++i)
    {
        const Call &call = canvas.at(i);
        if (call.kind == CALL_TEXT)
        {
            TEST_ASSERT_EQUAL_INT(SSD1306_HEADER_TILT_X, call.x0);
            TEST_ASSERT_EQUAL_STRING("+07/+00", call.text);
        }
    }
}

void test_invalidate_forces_a_full_redraw(void)
{
    RecordingCanvas canvas;
    DisplayRenderer<RecordingCanvas> renderer(canvas);
    const DisplayFrame frame = baseFrame();

    renderer.render(frame);
    renderer.invalidate();
    canvas.reset();

    TEST_ASSERT_TRUE(renderer.render(frame));
    TEST_ASSERT_EQUAL_INT(1, canvas.countOf(CALL_ERASE));
}

void test_extend_points_down_and_retract_points_up(void)
{
    const struct
    {
        ActuatorGlyph glyph;
        bool isWidestAtTop;
    } cases[] = {
        {ActuatorGlyph::Extend, true},
        {ActuatorGlyph::Retract, false},
    };

    for (size_t caseIndex = 0; caseIndex < 2; ++caseIndex)
    {
        RecordingCanvas canvas;
        DisplayFrame moved = baseFrame();
        setDisplayActuatorGlyph(
            moved,
            ActuatorId::FRHY,
            cases[caseIndex].glyph);

        renderThenCapture(canvas, baseFrame(), moved);

        int topY = 0, bottomY = 0, topWidth = -1, bottomWidth = -1;
        bool hasSeenSpan = false;
        for (int i = 0; i < canvas.count(); ++i)
        {
            const Call &call = canvas.at(i);
            if (call.kind != CALL_LINE || call.y0 != call.y1)
                continue;
            const int width = call.x1 - call.x0;
            if (width > SSD1306_GLYPH_SIZE)
                continue;
            if (!hasSeenSpan || call.y0 < topY)
            {
                topY = call.y0;
                topWidth = width;
            }
            if (!hasSeenSpan || call.y0 > bottomY)
            {
                bottomY = call.y0;
                bottomWidth = width;
            }
            hasSeenSpan = true;
        }

        TEST_ASSERT_TRUE(hasSeenSpan);
        if (cases[caseIndex].isWidestAtTop)
            TEST_ASSERT_GREATER_THAN_INT(bottomWidth, topWidth);
        else
            TEST_ASSERT_GREATER_THAN_INT(topWidth, bottomWidth);
    }
}

void test_a_joint_clears_only_out_to_its_inboard_neighbour(void)
{
    const ActuatorId actuatorIds[][3] = {
        {ActuatorId::FLHY, ActuatorId::FLHL, ActuatorId::FLKL},
        {ActuatorId::FRHY, ActuatorId::FRHL, ActuatorId::FRKL},
    };
    for (size_t side = 0; side < 2; ++side)
    {
        int left[3];
        int right[3];

        for (size_t actuator = 0; actuator < 3; ++actuator)
        {
            RecordingCanvas canvas;
            DisplayFrame moved = baseFrame();
            setDisplayActuatorGlyph(
                moved,
                actuatorIds[side][actuator],
                ActuatorGlyph::Disconnected);

            renderThenCapture(canvas, baseFrame(), moved);

            int found = 0;
            for (int i = 0; i < canvas.count(); ++i)
            {
                const Call &call = canvas.at(i);
                if (call.kind != CALL_FILL || call.color != SSD1306_COLOR_BLACK)
                    continue;
                left[actuator] = call.x0;
                right[actuator] = call.x0 + call.x1 - 1;
                ++found;
            }
            TEST_ASSERT_EQUAL_INT(1, found);
        }

        for (size_t actuator = 1; actuator < 3; ++actuator)
        {
            const bool isLeftward = left[actuator] < left[actuator - 1];
            if (isLeftward)
                TEST_ASSERT_LESS_THAN_INT(left[actuator - 1], right[actuator]);
            else
                TEST_ASSERT_GREATER_THAN_INT(right[actuator - 1], left[actuator]);
        }
    }
}

void test_a_battery_change_redraws_only_that_battery(void)
{
    RecordingCanvas canvas;
    DisplayFrame drained = baseFrame();
    drained.batteryLevel[1] = 0.1f;

    renderThenCapture(canvas, baseFrame(), drained);

    const int cellX = SSD1306_BATTERY_CELL_X + SSD1306_BATTERY_CELL_PITCH;
    TEST_ASSERT_EQUAL_INT(1, canvas.countOf(CALL_RECT));
    for (int i = 0; i < canvas.count(); ++i)
    {
        const Call &call = canvas.at(i);
        if (call.kind == CALL_RECT)
        {
            TEST_ASSERT_EQUAL_INT(cellX + SSD1306_BATTERY_BAR_DX, call.x0);
            TEST_ASSERT_EQUAL_INT(SSD1306_BATTERY_Y, call.y0);
        }
    }
}

void test_each_gauge_is_labelled_and_shows_its_voltage(void)
{
    const struct
    {
        int16_t decivolts;
        const char *label;
        const char *voltage;
    } cases[] = {
        {134, "A", "13.4V"},
        {127, "B", "12.7V"},
        {0, "B", " 0.0V"},
    };

    for (size_t caseIndex = 0; caseIndex < 3; ++caseIndex)
    {
        const int battery = cases[caseIndex].label[0] - 'A';
        RecordingCanvas canvas;
        DisplayFrame changed = baseFrame();
        changed.batteryDecivolts[battery] = cases[caseIndex].decivolts;

        renderThenCapture(canvas, baseFrame(), changed);

        const int cellX =
            SSD1306_BATTERY_CELL_X + battery * SSD1306_BATTERY_CELL_PITCH;
        bool hasSeenLabel = false;
        bool hasSeenVoltage = false;
        for (int i = 0; i < canvas.count(); ++i)
        {
            const Call &call = canvas.at(i);
            if (call.kind != CALL_TEXT)
                continue;
            if (call.x0 == cellX && strcmp(call.text, cases[caseIndex].label) == 0)
                hasSeenLabel = true;
            if (call.x0 == cellX + SSD1306_BATTERY_VALUE_DX &&
                strcmp(call.text, cases[caseIndex].voltage) == 0)
                hasSeenVoltage = true;
        }
        TEST_ASSERT_TRUE(hasSeenLabel);
        TEST_ASSERT_TRUE(hasSeenVoltage);
    }
}

void test_a_gauge_clears_only_its_own_cell(void)
{
    for (int battery = 0; battery < 2; ++battery)
    {
        RecordingCanvas canvas;
        DisplayFrame changed = baseFrame();
        changed.batteryLevel[battery] = 0.25f;

        renderThenCapture(canvas, baseFrame(), changed);

        const int cellX =
            SSD1306_BATTERY_CELL_X + battery * SSD1306_BATTERY_CELL_PITCH;
        int clears = 0;
        for (int i = 0; i < canvas.count(); ++i)
        {
            const Call &call = canvas.at(i);
            if (call.kind != CALL_FILL || call.color != SSD1306_COLOR_BLACK)
                continue;
            ++clears;
            TEST_ASSERT_EQUAL_INT(cellX, call.x0);
            TEST_ASSERT_TRUE(call.x1 <= SSD1306_BATTERY_CELL_PITCH);
        }
        TEST_ASSERT_EQUAL_INT(1, clears);
    }
}

void test_invalid_actuator_has_no_display_position(void)
{
    TEST_ASSERT_EQUAL_INT(0, ssd1306ActuatorX(ActuatorId::ActuatorCount));
    TEST_ASSERT_EQUAL_INT(0, ssd1306ActuatorY(ActuatorId::ActuatorCount));
}

void test_a_controller_presence_change_redraws_the_body(void)
{
    RecordingCanvas canvas;
    DisplayFrame changed = baseFrame();
    changed.controllers[ROLE_RIGHT] = false;

    renderThenCapture(canvas, baseFrame(), changed);

    TEST_ASSERT_TRUE(canvas.countOf(CALL_FILL) > 0);
    TEST_ASSERT_TRUE(canvas.countOf(CALL_RECT) > 0);
}

void test_missing_battery_readings_replace_voltage_labels(void)
{
    RecordingCanvas canvas;
    DisplayFrame measured = baseFrame();
    const Volts readings[2] = {Volts(13.4f), Volts(12.0f)};
    setBatteryVoltages(measured, readings);
    renderThenCapture(canvas, measured, DisplayFrame{});
    int missingLabels = 0;
    for (int i = 0; i < canvas.count(); ++i)
        if (canvas.at(i).kind == CALL_TEXT && strcmp(canvas.at(i).text, "--.-V") == 0)
            ++missingLabels;
    TEST_ASSERT_EQUAL_INT(3, missingLabels);
    DisplayRenderer<RecordingCanvas> renderer(canvas);
    renderer.render(DisplayFrame{});
    canvas.reset();
    TEST_ASSERT_FALSE(renderer.render(DisplayFrame{}));
    TEST_ASSERT_EQUAL_INT(0, canvas.count());
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_missing_battery_readings_replace_voltage_labels);
    RUN_TEST(test_the_first_frame_erases_and_rules_the_header);
    RUN_TEST(test_an_unchanged_model_draws_nothing);
    RUN_TEST(test_a_tilt_change_redraws_only_the_tilt_field);
    RUN_TEST(test_invalidate_forces_a_full_redraw);
    RUN_TEST(test_extend_points_down_and_retract_points_up);
    RUN_TEST(test_a_joint_clears_only_out_to_its_inboard_neighbour);
    RUN_TEST(test_a_battery_change_redraws_only_that_battery);
    RUN_TEST(test_each_gauge_is_labelled_and_shows_its_voltage);
    RUN_TEST(test_a_gauge_clears_only_its_own_cell);
    RUN_TEST(test_invalid_actuator_has_no_display_position);
    RUN_TEST(test_a_controller_presence_change_redraws_the_body);
    return UNITY_END();
}
