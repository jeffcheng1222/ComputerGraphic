#include "control.h"
#include <glm/glm.hpp>
#include <utility>

namespace Control {

Pressed left_mouse = UP, right_mouse = UP, mid_mouse = UP;
glm::vec2 last_mouse, mouse;
bool last_mouse_valid = false;
bool running = false, lastFrame = false;
bool moving = false;

glm::vec2 updateMousePos(glm::vec2 new_mouse)
{
	if (!last_mouse_valid) {
		last_mouse_valid = true;
		last_mouse = mouse = new_mouse;
	}
	else {
		last_mouse = mouse;
		mouse = new_mouse;
	}

	return mouse - last_mouse;
}

std::pair<glm::vec2, glm::vec2> getLastTwoMouse() {
    return { last_mouse, mouse };
}

glm::vec2 getMouseDiff() {
	return mouse - last_mouse;
}

}