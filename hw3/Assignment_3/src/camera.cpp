#include "camera.h"
#include "control.h"
#include "Helpers.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/common.hpp>
#include <glm/gtx/rotate_vector.hpp>

Camera* Camera::_instance = nullptr;

Camera& Camera::getInstance() {
	if (!_instance)
		fexit(-1, "Uninitialized camera");
	return *_instance;
}

Camera& Camera::newInstance(const glm::vec3 &pos, const glm::vec3 &focus, float aspect) {
	_instance = new Camera(pos, focus, aspect);
	return *_instance;
}

Camera::Camera()
{
}

Camera::Camera(const glm::vec3 & pos, const glm::vec3 &focus, float aspect)
{
	glm::vec3 front(focus - pos), up = glm::vec3(0.f, 1.f, 0.f);
	up = glm::vec3(0, 1, 0);
	*this = Camera(pos, front, up, 60.f, aspect);
}

Camera::Camera(const glm::vec3 & pos, const glm::vec3 front, const glm::vec3 up, float fov, float aspect)
	: pos(pos)
	, front(front)
	, up(glm::normalize(up))
	, fov(fov)
	, aspect(aspect)
{
	rotx = glm::vec3(0.f, 1.f, 0.f);
	roty = glm::normalize(glm::cross(front, rotx));
	mode = PERSPECTIVE;
	depth_p2o = glm::length(pos);
}


Camera::~Camera()
{
}

void Camera::use(const Shader & shader, bool translate_invariant) const
{
	glm::mat4 view = glm::lookAt(pos, pos + front, up);
	if (translate_invariant) {
		view = glm::mat4(glm::mat3(view));
	}
	glm::mat4 pers;
	if (mode == PERSPECTIVE)
		pers = glm::perspective(glm::radians(fov), aspect, 0.1f, 100.f);
	else {
		ProjectionInfo info = getProjectionInfo();
		float scale = depth_p2o / info.n;
		pers = glm::ortho(scale * info.l, scale * info.r, scale * info.b, scale * info.t, 0.1f, 100.f);
	}
	shader.setUnif("view", view);
	shader.setUnif("proj", pers);
}

void Camera::setUp(const glm::vec3 &up_) { up = up_; }
void Camera::setPos(const glm::vec3 &pos_) { pos = pos_; }
void Camera::setFront(const glm::vec3 &front_) { front = front_; }

void Camera::setAspect(float aspect_)
{
	aspect = aspect_;
}

void Camera::rotate(const glm::vec2 dxy) {

	glm::vec3 center = pos + front;
	/* For horizontal panning, rotate camera within plane perpendicular to `up' direction */
	if (dxy.x != 0) {
		const glm::vec3 &axis = rotx;
		/* for now, manually update pos, front and up in renderer */
		front = glm::rotate(front, -dxy.x * Control::SCREEN_ROTATE_RATE, axis);
		up = glm::rotate(up, -dxy.x * Control::SCREEN_ROTATE_RATE, axis);
		pos = center - front;

		roty = glm::rotate(roty, -dxy.x * Control::SCREEN_ROTATE_RATE, axis);
	}
	/* For verticle panning, rotate camera within plane perpendicular to cross(up, front) direction */
	if (dxy.y != 0) {
		const glm::vec3 &axis = roty;

		front = glm::rotate(front, -dxy.y * Control::SCREEN_ROTATE_RATE, axis);
		up = glm::rotate(up, -dxy.y * Control::SCREEN_ROTATE_RATE, axis);
		pos = center - front;
	}

}

void Camera::pan(const glm::vec2 dxy) {
	glm::vec3 cam_d = dxy.x * -glm::normalize(glm::cross(front, up)) + dxy.y * glm::normalize(up);
	pos += Control::SCREEN_PAN_RATE * cam_d * glm::length(front);
}

void Camera::translate(const glm::vec3 dxyz)
{
	glm::vec3 focus = pos + front;
	pos += dxyz;
	front = focus - pos;
}

void Camera::zoom(float dy)
{
	const float min_d = 0.1f, max_d = 10.f;
	if (dy > 0) {
		if (front.length() < min_d) return;
		pos += front * Control::SCREEN_SCROLL_RATE;
		front -= front * Control::SCREEN_SCROLL_RATE;
	}
	else {
		if (front.length() > max_d) return;
		pos -= front * Control::SCREEN_SCROLL_RATE;
		front += front * Control::SCREEN_SCROLL_RATE;
	}

	depth_p2o = glm::length(pos);
}

ProjectionInfo Camera::getProjectionInfo() const
{
	ProjectionInfo i;
	float tanHalfFov = tan(glm::radians(fov) * 0.5f);
	i.n = 0.1f;
	i.f = 100.f;
	i.t = tanHalfFov * i.n;
	i.b = -i.t;
	i.r = aspect * i.t;
	i.l = -i.r;
	return i;
}

glm::mat4 Camera::getInverseView() const
{
	glm::mat4 view = glm::lookAt(pos, pos + front, up);
	return glm::inverse(view);
}

glm::vec3 Camera::screenToWorld(float x, float y, int win_width, int win_height) const {
    ProjectionInfo projInfo = getProjectionInfo();
    glm::mat4 invView = getInverseView();
    glm::vec4 ePos((2.*x / win_width - 1) * projInfo.r, -(2.*y / win_height - 1) * projInfo.t, -projInfo.n, 1);
    return glm::vec3(invView * ePos);
}

void Camera::screenToRay(float x, float y, int win_width, int win_height, glm::vec3& origin, glm::vec3 &dir) const
{
	ProjectionInfo projInfo = getProjectionInfo();
	if (mode == PERSPECTIVE) {
		origin = getPos();
		glm::mat4 invView = getInverseView();
		glm::vec4 ePos((2. * x / win_width - 1) * projInfo.r, -(2. * y / win_height - 1) * projInfo.t, -projInfo.n, 1);
		dir = glm::normalize(glm::vec3(invView * ePos) - origin);
	}
	else {
		float scale = depth_p2o / projInfo.n;
		glm::mat4 invView = getInverseView();
		glm::vec4 ePos((2. * x / win_width - 1) * projInfo.r, -(2. * y / win_height - 1) * projInfo.t, -projInfo.n, 1);
		glm::vec4 oPos((2. * x / win_width - 1) * projInfo.r, -(2. * y / win_height - 1) * projInfo.t, 0, 1);
		ePos *= scale;
		oPos *= scale;
		origin = glm::vec3(invView * oPos);
		dir = glm::normalize(glm::vec3(invView * ePos) - origin);
	}
}

glm::vec3 Camera::mouseDiffToWorld(const glm::vec2 last, const glm::vec2 now, int win_width, int win_height) const
{
    return screenToWorld(now.x, now.y, win_width, win_height) -
        screenToWorld(last.x, last.y, win_width, win_height);
}
