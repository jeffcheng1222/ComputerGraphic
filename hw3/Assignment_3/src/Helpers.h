#ifndef HELPER_H
#define HELPER_H

#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glm/glm.hpp>

#ifdef _WIN32
#  include <windows.h>
#  undef max
#  undef min
#  undef DrawText
#endif

#ifndef __APPLE__
#  define GLEW_STATIC
#  include <GL/glew.h>
#endif

#ifdef __APPLE__
#   include <OpenGL/gl3.h>
#   define __gl_h_ /* Prevent inclusion of the old gl.h */
#else
#   ifdef _WIN32
#       include <windows.h>
#   endif
#   include <GL/gl.h>
#endif

#include <GLFW/glfw3.h>


class VertexArrayObject
{
public:
    unsigned int id;

    VertexArrayObject() : id(0) {}

    // Create a new VAO
    void init();

    // Select this VAO for subsequent draw calls
    void bind();

    // Release the id
    void free();
};

class VertexBufferObject
{
public:
    typedef unsigned int GLuint;
    typedef int GLint;

    GLuint id;
    GLuint rows;
    GLuint cols;

    VertexBufferObject() : id(0), rows(0), cols(0) {}

    // Create a new empty VBO
    void init();

    // Updates the VBO with a matrix M
    void update(const Eigen::MatrixXf& M);

    // Select this VBO for subsequent draw calls
    void bind();

    // Release the id
    void free();
};

struct Ray {
	glm::vec3 origin, dir;
};

struct Object {
	GLuint vao, vbo_pos, vbo_normal, ebo;
	Eigen::Affine3f translate, scale, rotate;
	std::vector<std::vector<Eigen::Vector3f>> faces;
	int nfaces;
	bool selected;

	Object() { selected = false;  }
	Object(const char* filename) : selected(false) { load(filename); }

	void load(const char* filename);
	void draw() const;
	Eigen::Affine3f getTransform() const;
	float intersect(const Ray& ray) const;
};

extern Ray camRay;
void updateCamRay(int screen_x, int screen_y);

// This class wraps an OpenGL program composed of two shaders
class Program
{
public:
  typedef unsigned int GLuint;
  typedef int GLint;

  GLuint vertex_shader;
  GLuint fragment_shader;
  GLuint program_shader;

  Program() : vertex_shader(0), fragment_shader(0), program_shader(0) { }

  // Create a new shader from the specified source strings
  bool init(const std::string &vertex_shader_string,
  const std::string &fragment_shader_string,
  const std::string &fragment_data_name);

  // Select this shader for subsequent draw calls
  void bind();

  // Release all OpenGL objects
  void free();

  // Return the OpenGL handle of a named shader attribute (-1 if it does not exist)
  GLint attrib(const std::string &name) const;

  // Return the OpenGL handle of a uniform attribute (-1 if it does not exist)
  GLint uniform(const std::string &name) const;

  // Bind a per-vertex array attribute
  GLint bindVertexAttribArray(const std::string &name, VertexBufferObject& VBO) const;

  GLuint create_shader_helper(GLint type, const std::string &shader_string);

};

// From: https://blog.nobel-joergensen.com/2013/01/29/debugging-opengl-using-glgeterror/
void _check_gl_error(const char *file, int line);

void fexit(const int code, const char* msg);

void get_window_size(GLFWwindow* window, int& width, int& height);

///
/// Usage
/// [... some opengl calls]
/// glCheckError();
///
#define check_gl_error() _check_gl_error(__FILE__,__LINE__)

#endif
