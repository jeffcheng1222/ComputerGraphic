#include "Helpers.h"
#include "camera.h"

#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <climits>
#include <string>
#include <vector>
#include <glm/glm.hpp>
using namespace std;

void _check_gl_error(const char *file, int line)
{
  GLenum err (glGetError());

  while(err!=GL_NO_ERROR)
  {
    std::string error;

    switch(err)
    {
      case GL_INVALID_OPERATION:      error="INVALID_OPERATION";      break;
      case GL_INVALID_ENUM:           error="INVALID_ENUM";           break;
      case GL_INVALID_VALUE:          error="INVALID_VALUE";          break;
      case GL_OUT_OF_MEMORY:          error="OUT_OF_MEMORY";          break;
      case GL_INVALID_FRAMEBUFFER_OPERATION:  error="INVALID_FRAMEBUFFER_OPERATION";  break;
    }

    std::cerr << "GL_" << error.c_str() << " - " << file << ":" << line << std::endl;
    err = glGetError();
  }
}

void fexit(const int code, const char* msg) {
	if (msg)
		fprintf(stderr, msg);
	return exit(code);
}

void get_window_size(GLFWwindow* window, int& width, int& height) {
	glfwGetWindowSize(window, &width, &height);
}

void Object::load(const char* filename)
{
	static char sbuf[1024];
	FILE* fp = fopen(filename, "r");

	if (!fp)
		fexit(-1, (string("Cannot open ") + string(filename) + "\n").c_str());

	int nv, nf, ne;
	fscanf(fp, "%s%d%d%d", sbuf, &nv, &nf, &ne);
	
	glm::fvec3 aabb_min(FLT_MAX), aabb_max(FLT_MIN), aabb_center, aabb_scale;
	vector<glm::fvec3> vertices(nv), fnormals(nf), vnormals(nv);
	vector<glm::ivec3> indices(nf);
	vector<vector<int>> vfaces(nv);

	// TODO: check if component wise min/max is OK.

	// Read file
	for (int i = 0; i < nv; i++) {
		fscanf(fp, "%f%f%f", &vertices[i].x, &vertices[i].y, &vertices[i].z);
		aabb_min.x = min(vertices[i].x, aabb_min.x);
		aabb_min.y = min(vertices[i].y, aabb_min.y);
		aabb_min.z = min(vertices[i].z, aabb_min.z);
		aabb_max.x = max(vertices[i].x, aabb_max.x);
		aabb_max.y = max(vertices[i].y, aabb_max.y);
		aabb_max.z = max(vertices[i].z, aabb_max.z);
	}

	aabb_center = 0.5f * (aabb_min + aabb_max);
	aabb_scale = aabb_max - aabb_min;

	for (int i = 0; i < nv; i++)
		vertices[i] = (vertices[i] - aabb_center) / aabb_scale;

	for (int i = 0; i < nf; i++) {
		int m;
		fscanf(fp, "%d%d%d%d", &m, &indices[i].x, &indices[i].y, &indices[i].z);
		fnormals[i] = glm::cross(vertices[indices[i].y] - vertices[indices[i].x], vertices[indices[i].z] - vertices[indices[i].x]);
		fnormals[i] = glm::normalize(fnormals[i]);
		vfaces[indices[i].x].push_back(i);
		vfaces[indices[i].y].push_back(i);
		vfaces[indices[i].z].push_back(i);
		faces.push_back({ 
			Eigen::Vector3f(vertices[indices[i].x].x, vertices[indices[i].x].y, vertices[indices[i].x].z),
			Eigen::Vector3f(vertices[indices[i].y].x, vertices[indices[i].y].y, vertices[indices[i].y].z),
			Eigen::Vector3f(vertices[indices[i].z].x, vertices[indices[i].z].y, vertices[indices[i].z].z),
		});
	}

	// Calculate per-vertex normal
	for (int i = 0; i < nv; i++) {
		glm::fvec3 normal;
		for (int j : vfaces[i])
			normal += fnormals[j];
		if (vfaces[i].size() > 0) {
			normal /= vfaces[i].size();
			normal = glm::normalize(normal);
		}
		vnormals[i] = normal;
	}

	// Prepare VAO, VBO, EBO
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo_pos);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::fvec3), vertices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::fvec3), (void*)0);
	glEnableVertexAttribArray(0);

	glGenBuffers(1, &vbo_normal);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_normal);
	glBufferData(GL_ARRAY_BUFFER, vnormals.size() * sizeof(glm::fvec3), vnormals.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::fvec3), (void*)0);
	glEnableVertexAttribArray(1);

	glGenBuffers(1, &ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(glm::ivec3), indices.data(), GL_STATIC_DRAW);

	glBindVertexArray(0);

	// Initialize transformation
	translate = Eigen::Affine3f::Identity();
	scale = Eigen::Affine3f::Identity();
	rotate = Eigen::Affine3f::Identity();

	translate *= Eigen::Translation3f(0, 0, 0);
	scale *= Eigen::Scaling(1.f);
	rotate = Eigen::AngleAxisf(0 / 8, Eigen::Vector3f(1, 1, 1));

	nfaces = nf;
}

void Object::draw() const
{
	glBindVertexArray(vao);
	glDrawElements(GL_TRIANGLES, nfaces * 3, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}

Eigen::Affine3f Object::getTransform() const
{
	return translate * scale * rotate;
}

float Object::intersect(const Ray& ray) const
{
	const float kEpsilon = 1e-5f;
	const Eigen::Affine3f trans = getTransform();
	Eigen::Vector3f dir = Eigen::Vector3f(ray.dir.x, ray.dir.y, ray.dir.z), origin = Eigen::Vector3f(ray.origin.x, ray.origin.y, ray.origin.z);
	float mint = -1.f;
	for (auto& face : faces) {
		const Eigen::Vector3f v0 = trans * face[0], v1 = trans * face[1], v2 = trans * face[2];
		Eigen::Vector3f v0v1 = v1 - v0;
		Eigen::Vector3f v0v2 = v2 - v0;
		Eigen::Vector3f pvec = dir.cross(v0v2);
		float det = v0v1.dot(pvec);
		if (fabs(det) < kEpsilon) continue;
		float invDet = 1 / det;

		Eigen::Vector3f tvec = origin - v0;
		float u = tvec.dot(pvec) * invDet;
		if (u < 0 || u > 1) continue;

		Eigen::Vector3f qvec = tvec.cross(v0v1);
		float v = dir.dot(qvec) * invDet;
		if (v < 0 || u + v > 1) continue;

		float t = v0v2.dot(qvec) * invDet;
		if (mint == -1.f || t < mint) mint = t;
	}

	return mint;
}

Ray camRay;
extern GLFWwindow* window;

void updateCamRay(int screen_x, int screen_y) {
	int win_width, win_height;

	get_window_size(window, win_width, win_height);

	Camera& cam = Camera::getInstance();

	/*glm::vec3 camPos = cam.getPos();
	glm::vec3 cursorPos = cam.screenToWorld(screen_x, screen_y, win_width, win_height);
	glm::vec3 ray = cursorPos - camPos;

	camRay.dir = glm::normalize(ray);
	camRay.origin = camPos;*/

	cam.screenToRay(screen_x, screen_y, win_width, win_height, camRay.origin, camRay.dir);
}