// This example is heavily based on the tutorial at https://open.gl

#include <stdlib.h>
#include <stdio.h>
#include <vector>
using namespace std;

#include "Helpers.h"
#include "camera.h"
#include "control.h"

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
// GLFW is necessary to handle the OpenGL context
#include <GLFW/glfw3.h>
#else
// GLFW is necessary to handle the OpenGL context
#include <GLFW/glfw3.h>
#endif

#include <glm/glm.hpp>

// Linear Algebra Library
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

// Timer
#include <chrono>

GLFWwindow* window;

const char* mesh_vs =
"# version 330 core\n"
"\n"
"layout(location = 0) in vec3 aPos;\n"
"layout(location = 1) in vec3 aNormal;\n"
"\n"
"uniform mat4 view;\n"
"uniform mat4 proj;\n"
"uniform mat4 model;"
"\n"
"out vec4 vNormal;\n"
"out vec3 vPos;"
"\n"
"void main() {\n"
"	gl_Position = proj * view * model * vec4(aPos, 1);\n"
"	vNormal = view * vec4(aNormal, 0);\n"
"   vPos = (view * model * vec4(aPos, 1)).xyz;"
"}\n";

const char* mesh_fs =
"# version 330 core\n"
"uniform bool isflat;"
"uniform bool iswireframe;"
"uniform bool selected;"
"in vec4 vNormal;"
"in vec3 vPos;"
"out vec4 FragColor;"
""
"void main() {"
"   vec3 lightPos = vec3(0,0,10);"
"   vec3 diff = normalize(lightPos-vPos);"
"   vec3 normal = vNormal.xyz;"
"   if (isflat && !iswireframe) {"
"   vec3 xTan = dFdx(vPos), yTan = dFdy(vPos); normal.xyz = normalize(cross(xTan, yTan));"
"	}"	
"   vec3 reflectDiff = reflect(diff, normal);"
"   float spec = 0.5 * pow(max(dot(normalize(lightPos), -reflectDiff), 0.0), 256);"
"   float amb = 0.4;"
"	float c = 0.5 * dot(diff, normal) + spec + amb ;"
"   vec3 base = selected ? vec3(0.9, 0.4, 0.4) : vec3(0.4, 0.9, 0.6);"
"	FragColor = vec4(c * base, 1);"
"}";

Shader mesh_shader;
bool is_flat = true;
bool is_wireframe = false;

Object cube;
vector<Object> objects;

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	Camera::getInstance().setAspect(1.f * width / height);
    glViewport(0, 0, width, height);
}

void mouse_move_callback(GLFWwindow* window, double xpos, double ypos)
{
	Control::updateMousePos(glm::vec2(xpos, ypos));
	updateCamRay(xpos, ypos);
	/* -- Camera control -- */

   /* Rotating */
	Camera& cam = Camera::getInstance();
	glm::vec2 scr_d = Control::getMouseDiff();

	if (Control::left_mouse == Control::DOWN)
		cam.rotate(scr_d);

	/* Panning (Disabled) */
	if (false && Control::right_mouse == Control::DOWN)
		cam.pan(scr_d);
}

int selectedObjectInd = -1;
void clickObject()
{
	if (selectedObjectInd != -1)
		objects[selectedObjectInd].selected = false;

	selectedObjectInd = -1;

	float mint = FLT_MAX;
	for (int i = 0; i < objects.size(); i++) {
		Object& obj = objects[i];
		float t = obj.intersect(camRay);
		if (t > 0 && t < mint) {
			mint = t;
			selectedObjectInd = i;
		}
	}

	if (selectedObjectInd != -1)
		objects[selectedObjectInd].selected = true;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	Control::updateMousePos(glm::vec2(xpos, ypos));
	Control::Pressed pressed = action == GLFW_PRESS ? Control::DOWN : Control::UP;

	if (button == GLFW_MOUSE_BUTTON_LEFT)
		Control::left_mouse = pressed;
	if (button == GLFW_MOUSE_BUTTON_RIGHT)
		Control::right_mouse = pressed;
	if (button == GLFW_MOUSE_BUTTON_MIDDLE)
		Control::mid_mouse = pressed;

	if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_LEFT)
		clickObject();
}

void mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	if (yoffset > 0)
		Camera::getInstance().zoom(1);
	else
		Camera::getInstance().zoom(-1);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (action == GLFW_PRESS)
		return;

    // Update the position of the first vertex if the keys 1,2, or 3 are pressed
	static Object idle;
	float sgn = (mods & GLFW_MOD_SHIFT) > 0 ? -1 : 1;
	Object& obj = selectedObjectInd == -1 ? idle : objects[selectedObjectInd];

    switch (key)
    {
	case GLFW_KEY_J:
		obj.rotate *= Eigen::AngleAxisf(sgn * glm::pi<float>() / 16.f, Eigen::Vector3f(1, 0, 0));
		break;
	case GLFW_KEY_K:
		obj.rotate *= Eigen::AngleAxisf(sgn * glm::pi<float>() / 16.f, Eigen::Vector3f(0, 1, 0));
		break;
	case GLFW_KEY_L:
		obj.rotate *= Eigen::AngleAxisf(sgn * glm::pi<float>() / 16.f, Eigen::Vector3f(0, 0, 1));
		break;
	case GLFW_KEY_U:
		obj.translate *= Eigen::Translation3f(sgn * 0.2, 0, 0);
		break;
	case GLFW_KEY_I:
		obj.translate *= Eigen::Translation3f(0, sgn * 0.2, 0);
		break;
	case GLFW_KEY_O:
		obj.translate *= Eigen::Translation3f(0, 0, sgn * 0.2);
		break;
	case GLFW_KEY_N:
		obj.scale *= Eigen::Scaling(1.2f);
		break;
	case GLFW_KEY_M:
		obj.scale *= Eigen::Scaling(1 / 1.2f);
		break;
	case GLFW_KEY_Z:
		is_flat = !is_flat;
		break;
	case GLFW_KEY_X:
		is_wireframe = !is_wireframe;
		break;
	case GLFW_KEY_C:
		Camera::getInstance().switchMode();
		break;
	case GLFW_KEY_1:
		objects.push_back(Object("data/cube.off"));
		break;
	case GLFW_KEY_3:
		objects.push_back(Object("data/bunny.off"));
		break;
	case GLFW_KEY_2:
		objects.push_back(Object("data/bunny_cube.off"));
		break;
	case GLFW_KEY_W:
		Camera::getInstance().translate(glm::vec3(0, 0.5, 0));
		break;
	case GLFW_KEY_S:
		Camera::getInstance().translate(glm::vec3(0, -0.5, 0));
		break;
	case GLFW_KEY_A:
		Camera::getInstance().translate(glm::vec3(-0.5, 0, 0));
		break;
	case GLFW_KEY_D:
		Camera::getInstance().translate(glm::vec3(0.5, 0, 0));
		break;
	default:
        break;
    }

}

void renderInit() {
	int win_width, win_height;
	get_window_size(window, win_width, win_height);
	// Camera
	const float cam_R = 6, cam_theta = glm::pi<float>() / 6.f, cam_phi = glm::pi<float>() / 6.f;
	const float cam_aspect = (float)win_width / win_height;
	// const glm::vec3 cam_pos(cam_R * cos(cam_phi) * cos(cam_theta), cam_R * sin(cam_phi) * cos(cam_theta), cam_R * sin(cam_theta));
	const glm::vec3 cam_pos(0, 0, 3);
	const glm::vec3 cam_lookat(0, 0, 0);
	Camera::newInstance(cam_pos, cam_lookat, cam_aspect);

	mesh_shader = Shader(mesh_vs, mesh_fs);
	cube = Object("data/cube.off");
}

int main(void)
{

    // Initialize the library
    if (!glfwInit())
        return -1;

    // Activate supersampling
    glfwWindowHint(GLFW_SAMPLES, 8);

    // Ensure that we get at least a 3.2 context
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);

    // On apple we have to load a core profile with forward compatibility
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    #ifndef __APPLE__
      glewExperimental = true;
      GLenum err = glewInit();
      if(GLEW_OK != err)
      {
        /* Problem: glewInit failed, something is seriously wrong. */
       fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
      }
      glGetError(); // pull and savely ignonre unhandled errors like GL_INVALID_ENUM
      fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));
    #endif

    int major, minor, rev;
    major = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR);
    minor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR);
    rev = glfwGetWindowAttrib(window, GLFW_CONTEXT_REVISION);
    printf("OpenGL version recieved: %d.%d.%d\n", major, minor, rev);
    printf("Supported OpenGL is %s\n", (const char*)glGetString(GL_VERSION));
    printf("Supported GLSL is %s\n", (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

	renderInit();

	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_MULTISAMPLE);
	glLineWidth(2);

    // Save the current time --- it will be used to dynamically change the triangle color
    auto t_start = std::chrono::high_resolution_clock::now();

    // Register the keyboard callback
    glfwSetKeyCallback(window, key_callback);

    // Register the mouse callback
    glfwSetMouseButtonCallback(window, mouse_button_callback);

	glfwSetScrollCallback(window, mouse_scroll_callback);

    // Update viewport
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	// Register the mouse move callback
	glfwSetCursorPosCallback(window, mouse_move_callback);

	objects = { cube };

	auto timer = std::chrono::system_clock::now();
	float deltaTime = 0;
    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window))
    {
        // Set the uniform value depending on the time difference
        auto t_now = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration_cast<std::chrono::duration<float>>(t_now - t_start).count();

		glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_TRUE);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glPolygonMode(GL_FRONT_AND_BACK, is_wireframe ? GL_LINE : GL_FILL);

		mesh_shader.use();
		Camera::getInstance().use(Shader::now());
		mesh_shader.setUnif("isflat", is_flat);
		mesh_shader.setUnif("iswireframe", is_wireframe);

		for (auto& object : objects) {
			mesh_shader.setUnif("model", object.getTransform());
			mesh_shader.setUnif("selected", object.selected);
			object.draw();
		}

        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();

		auto ttimer = std::chrono::system_clock::now();
		deltaTime = std::chrono::duration<float>(ttimer - timer).count();
		timer = ttimer;
    }


    // Deallocate glfw internals
    glfwTerminate();
    return 0;
}
