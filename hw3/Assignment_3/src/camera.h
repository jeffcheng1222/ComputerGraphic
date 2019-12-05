#ifndef __CAMERA_H_
#define __CAMERA_H_
#include "shader.h"
#include <glm/glm.hpp>

struct ProjectionInfo {
	float l, r;
	float t, b;
	float n, f;
};

class Camera
{
public:
	Camera();
	/* Specify only camera position and looks at origin */
	Camera(const glm::vec3 &pos, const glm::vec3 &focus, float aspect);
	Camera(const glm::vec3 &pos, const glm::vec3 front, const glm::vec3 up, float fov, float aspect);
	~Camera();

	static Camera& getInstance();
	static Camera& newInstance(const glm::vec3 &pos, const glm::vec3 &focus, float aspect);

	void switchMode() { mode = mode == PERSPECTIVE ? ORTHOGRAPHIC : PERSPECTIVE; }

	void use(const Shader &shader, bool translate_invariant = false) const;
	void setAspect(float aspect);
	void setPos(const glm::vec3 &pos);
	void setFront(const glm::vec3 &front);
	void setUp(const glm::vec3 &up);

	void rotate(const glm::vec2 dxy);
	void pan(const glm::vec2 dxy);
	void translate(const glm::vec3 dxyz);
	void zoom(float dy);

	const glm::vec3& getPos() const { return pos; }
	const glm::vec3& getUp() const { return up; }
	/* Convection: len(front) == len(lookat center - pos)
	 * or simply let front = lookat - pos
	 * This convection makes camera rotation work properly.
	 */
	const glm::vec3& getFront() const { return front; }

	ProjectionInfo getProjectionInfo() const;
	glm::mat4 getInverseView() const;
    glm::vec3 screenToWorld(float x, float y, int win_width, int win_height) const;
	void screenToRay(float x, float y, int win_width, int win_height, glm::vec3& origin, glm::vec3 &dir) const;
    glm::vec3 mouseDiffToWorld(const glm::vec2 last, const glm::vec2 now, int win_width, int win_height) const;

private:
	static Camera* _instance;

	glm::vec3 pos;
	glm::vec3 up;
	glm::vec3 front;

	/* Rotate axis when pan horizontally and vertically on screen */
	glm::vec3 rotx, roty;

	enum { PERSPECTIVE = 0, ORTHOGRAPHIC} mode;
	float fov;
	float aspect;
	float depth_p2o;
};

#endif