#ifndef __CONTROL_H_
#define __CONTROL_H_
#include <glm/glm.hpp>
#include <utility>

namespace Control {

const float SCREEN_ROTATE_RATE = 0.005f;
const float SCREEN_PAN_RATE = 0.002f;
const float SCREEN_SCROLL_RATE = 0.1f;

enum Pressed { UP=0, DOWN };

extern Pressed left_mouse, right_mouse, mid_mouse;
extern glm::vec2 last_mouse, mouse;
extern bool last_mouse_valid;
extern bool running, lastFrame;
extern unsigned int hlIndex;
extern int startMovingFrame;
extern bool moving;

glm::vec2 updateMousePos(glm::vec2 mouse);
std::pair<glm::vec2, glm::vec2> getLastTwoMouse();
glm::vec2 getMouseDiff();

}

#endif