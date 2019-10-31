// This example is heavily based on the tutorial at https://open.gl

#include <stdlib.h>
#include <stdio.h>

// OpenGL Helpers to reduce the clutter
#include "Helpers.h"

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
// GLFW is necessary to handle the OpenGL context
#include <GLFW/glfw3.h>
#else
// GLFW is necessary to handle the OpenGL context
#include <GLFW/glfw3.h>
#endif


// Linear Algebra Library
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

// Timer
#include <chrono>

enum State
{
	None,
	Insert,
	Translate,
	Delete,
	Color,
	Play
} state;

enum InsertState
{
	WaitForFirst,
	WaitForSecond,
	WaitForThird
} insertState;

enum TranslateState
{
	Released,
	Hold
} translateState;

// VertexBufferObject for vertex
VertexBufferObject VBO_V;
// VertexBufferObject for color
VertexBufferObject VBO_C;

// Contains the vertex positions
Eigen::MatrixXf V(2, 3), V1, V2;
Eigen::MatrixXf C(3, 3), C1, C2;
Eigen::MatrixXf E(2, 6), E1, E2;
Eigen::Affine2f View(Eigen::Translation2f(0.f, 0.f));

const Eigen::Vector3f Red = Eigen::Vector3f(1, 0, 0);
const Eigen::Vector3f Blue = Eigen::Vector3f(0, 0, 1);

int GetNumTriangles()
{
	return V.cols() / 3;
}

void BSetVertex(int triInd, int vInd, const Eigen::Vector2f &p)
{
	if (triInd < 0)
		triInd = GetNumTriangles() + triInd;

	if (vInd < 0)
		vInd = 3 + vInd;

	V.col(triInd * 3 + vInd) = p;

	if (vInd == 0)
		E.col(triInd * 6) = E.col(triInd * 6 + 5) = p;
	else
		E.col(triInd * 6 + 2 * vInd - 1) = E.col(triInd * 6 + 2 * vInd) = p;
}

void BSetVertexColor(int triInd, int vInd, const Eigen::Vector3f& c)
{
	if (triInd < 0)
		triInd = GetNumTriangles() + triInd;

	if (vInd < 0)
		vInd = 3 + vInd;

	C.col(triInd * 3 + vInd) = c;
}

void BAddTriangle(const Eigen::MatrixXf &tri)
{
	assert(tri.rows() == 2 && tri.cols() == 3);

	V.conservativeResize(V.rows(), V.cols() + 3);
	C.conservativeResize(C.rows(), C.cols() + 3);
	E.conservativeResize(E.rows(), E.cols() + 6);

	for (int i = 0; i < 3; i++) {
		BSetVertex(-1, i, tri.col(0));
		BSetVertexColor(-1, i, Red);
	}
}

void BAddTriangle(const Eigen::Vector2f &a, const Eigen::Vector2f &b, const Eigen::Vector2f &c)
{
	V.conservativeResize(V.rows(), V.cols() + 3);
	C.conservativeResize(C.rows(), C.cols() + 3);
	E.conservativeResize(E.rows(), E.cols() + 6);

	BSetVertex(-1, 0, a);
	BSetVertex(-1, 1, b);
	BSetVertex(-1, 2, c);

	BSetVertexColor(-1, 0, Red);
	BSetVertexColor(-1, 1, Red);
	BSetVertexColor(-1, 2, Red);
}

Eigen::MatrixXf BGetTriangle(int triInd)
{
	if (triInd < 0)
		triInd = GetNumTriangles() + triInd;

	return V.block<2, 3>(0, triInd * 3);
}

void BRemoveTriangle(int triInd)
{
	Eigen::MatrixXf nV(V.rows(), V.cols() - 3), nC(C.rows(), C.cols() - 3), nE(E.rows(), E.cols() - 6);

	nV << V.block(0, 0, V.rows(), triInd * 3), V.block(0, triInd * 3 + 3, V.rows(), V.cols() - triInd * 3 - 3);
	nC << C.block(0, 0, C.rows(), triInd * 3), C.block(0, triInd * 3 + 3, C.rows(), C.cols() - triInd * 3 - 3);
	nE << E.block(0, 0, E.rows(), triInd * 6), E.block(0, triInd * 6 + 6, E.rows(), E.cols() - triInd * 6 - 6);

	V = nV;
	C = nC;
	E = nE;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void insertHandleClick(const Eigen::Vector2f &worldP)
{
	if (insertState == WaitForFirst)
	{
		// Add one triangle
		BAddTriangle(worldP, worldP, worldP);
		insertState = WaitForSecond;
	}
	else if (insertState == WaitForSecond)
	{
		insertState = WaitForThird;
	}
	else
	{
		insertState = WaitForFirst;
	}
}

float sign(const Eigen::Vector2f& a, const Eigen::Vector2f& b, const Eigen::Vector2f& c)
{
	return (a.x() - c.x()) * (b.y() - c.y()) - (b.x() - c.x()) * (a.y() - c.y());
}

bool pointInTriangle(const Eigen::Vector2f& p, const Eigen::Vector2f& a, const Eigen::Vector2f& b, const Eigen::Vector2f& c)
{
	float d1, d2, d3;
	bool has_neg, has_pos;

	d1 = sign(p, a, b);
	d2 = sign(p, b, c);
	d3 = sign(p, c, a);

	has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
	has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

	return !(has_neg && has_pos);
}

int selectedTriInd = -1;
Eigen::Vector2f p_pressedMouse;
Eigen::MatrixXf p_selectedTriSrc;

int findTriIndByPos(const Eigen::Vector2f& p)
{
	int target = -1;
	for (int i = 0; i < GetNumTriangles(); i++)
	{
		Eigen::Vector2f a = V.col(i * 3), b = V.col(i * 3 + 1), c = V.col(i * 3 + 2);
		if (pointInTriangle(p, a, b, c))
		{
			target = i;
			break;
		}
	}

	return target;
}

void unselectTriangle()
{
	if (selectedTriInd == -1) return;

	for (int i = 0; i < 3; i++)
		BSetVertexColor(selectedTriInd, i, Red);

	selectedTriInd = -1;
}

void selectTriangle(int triInd)
{
	unselectTriangle();
	selectedTriInd = triInd;

	if (triInd != -1)
	{
		p_selectedTriSrc = BGetTriangle(triInd);
		for (int i = 0; i < 3; i++)
			BSetVertexColor(triInd, i, Blue);
	}
}

void translateHandleClick(int action, const Eigen::Vector2f &worldP)
{
	if (action == GLFW_PRESS)
	{
		assert(translateState == Released);
		translateState = Hold;

		p_pressedMouse = worldP;

		selectTriangle(findTriIndByPos(worldP));
	}
	else
	{
		assert(translateState == Hold);
		translateState = Released;
	}
}

int selectedVertex = -1;
void colorHandleClick(const Eigen::Vector2f& worldP)
{
	float min_dist2 = 1.e38;
	for (int i = 0; i < V.cols(); i++) 
	{
		Eigen::Vector2f diff = V.col(i) - worldP;
		float dist2 = diff.dot(diff);

		if (dist2 < min_dist2)
		{
			min_dist2 = dist2;
			selectedVertex = i;
		}
	}
}

void deleteHandleClick(const Eigen::Vector2f& worldP)
{
	int targetTriInd = findTriIndByPos(worldP);

	if (targetTriInd != -1)
	{
		BRemoveTriangle(targetTriInd);
	}
}

Eigen::Vector2f screenToWorld(GLFWwindow* window, double xpos, double ypos)
{
	// Get the size of the window
	int width, height;
	glfwGetWindowSize(window, &width, &height);

	// Convert screen position to world coordinates
	// NOTE: y axis is flipped in glfw
	Eigen::Vector2f p(((xpos / double(width)) * 2) - 1, (((height - 1 - ypos) / double(height)) * 2) - 1);
	p = View.inverse() * p;

	return p;
}

void mouse_move_callback(GLFWwindow* window, double xpos, double ypos)
{
	Eigen::Vector2f worldP = screenToWorld(window, xpos, ypos);

	if (state == Insert)
	{
		if (insertState == WaitForSecond)
		{
			BSetVertex(-1, 1, worldP);
			BSetVertex(-1, 2, worldP);
			VBO_V.update(V);
		}
		else if (insertState == WaitForThird)
		{
			BSetVertex(-1, 2, worldP);
			VBO_V.update(V);
		}
	}
	else if (state == Translate)
	{
		if (translateState == Hold)
		{
			Eigen::Vector2f mouseDiff = worldP - p_pressedMouse;
			if (selectedTriInd != -1)
			{
				for (int i = 0; i < 3; i++)
					BSetVertex(selectedTriInd, i, p_selectedTriSrc.col(i) + mouseDiff);
			}
		}
	}
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{

	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	Eigen::Vector2f p = screenToWorld(window, xpos, ypos);

	if (action == GLFW_PRESS) {
		if (state == Insert)
			insertHandleClick(p);
		else if (state == Translate)
			translateHandleClick(action, p);
		else if (state == Color)
			colorHandleClick(p);
	}
	else
	{
		if (state == Translate)
			translateHandleClick(action, p);
		else if (state == Delete)
			deleteHandleClick(p);
	}


	return;

    // Update the position of the first vertex if the left button is pressed
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
        V.col(0) << p;

    // Upload the change to the GPU
    VBO_V.update(V);
}

void enterInsertMode()
{
	state = Insert;
	insertState = WaitForFirst;

	printf("[Insert] On\n");
}

void enterTranslateMode()
{
	state = Translate;
	translateState = Released;

	printf("[Translate] On\n");
}

void enterDeleteMode()
{
	state = Delete;

	printf("[Delete] On\n");
}

void enterColorMode()
{
	state = Color;

	printf("[Color] On\n");
}

void toggleKeyframePlaying()
{
	if (state != Play)
	{
		state = Play;
		printf("[Play] On\n");
	}
	else
	{
		state = None;
		printf("[Play] Off\n");
	}
}

void handleRotate(int key)
{
	if (selectedTriInd == -1) return;

	float degree = key == GLFW_KEY_H ? 10 : -10;
	float radius = degree / 180. * EIGEN_PI;
	Eigen::Rotation2Df rot(radius);
	Eigen::Matrix<float, 2, -1> triangle = BGetTriangle(selectedTriInd);
	Eigen::Vector2f center = (triangle.col(0) + triangle.col(1) + triangle.col(2)) / 3.;
	Eigen::Translation2f offset(center);
	Eigen::Affine2f rotTrans = offset * rot * offset.inverse();

	for (int i = 0; i < 3; i++)
		BSetVertex(selectedTriInd, i, rotTrans * triangle.col(i));
}

void handleScale(int key)
{
	if (selectedTriInd == -1) return;

	float scale = key == GLFW_KEY_K ? 1.2f : 1 / 1.2f;
	Eigen::Matrix<float, 2, -1> triangle = BGetTriangle(selectedTriInd);
	Eigen::Vector2f center = (triangle.col(0) + triangle.col(1) + triangle.col(2)) / 3.;
	Eigen::Translation2f offset(center);
	Eigen::Affine2f scalingTrans = offset * Eigen::Scaling(scale) * offset.inverse();

	for (int i = 0; i < 3; i++)
		BSetVertex(selectedTriInd, i, scalingTrans * triangle.col(i));
}

void captureKeyframe(int key)
{
	auto& kV = key == GLFW_KEY_T ? V1 : V2;
	auto& kC = key == GLFW_KEY_T ? C1 : C2;
	auto& kE = key == GLFW_KEY_T ? E1 : E2;

	kV = V; 
	kC = C;
	kE = E;

	printf("[Play] Capture Keyframe %d\n", key == GLFW_KEY_T ? 0 : 1);	
}

void handleColorBtn(int key)
{
	if (state != Color || selectedVertex == -1) return;

	const Eigen::Vector3f colors[] = {
		Eigen::Vector3f(255, 87, 34) / 255,
		Eigen::Vector3f(217, 203, 158) / 255,
		Eigen::Vector3f(55, 65, 64) / 255,
		Eigen::Vector3f(0, 163, 136) / 255,
		Eigen::Vector3f(22, 128, 57) / 255,
		Eigen::Vector3f(217, 0, 0) / 255,
		Eigen::Vector3f(115, 45, 217) / 255,
		Eigen::Vector3f(156, 39, 176) / 255,
		Eigen::Vector3f(242, 183, 5) / 255
	};

	int colorInd = key - GLFW_KEY_1;
	C.col(selectedVertex) = colors[colorInd];
}

char exportPathBuf[4096];
void exportSVG(GLFWwindow *window)
{
	int w_width, w_height;
	glfwGetWindowSize(window, &w_width, &w_height);

	FILE* f = fopen("export.svg", "w");
#ifdef _WIN32
	char* fullpath = _fullpath(exportPathBuf, "./export.svg", 4096);
#else
	char* fullpath = realpath("./export.svg", exportPathBuf);
#endif
	fprintf(f, "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n");
	fprintf(f, "<svg width=\"%d\" height=\"%d\" viewBox=\"%d %d %d %d\" xmlns=\"http://www.w3.org/2000/svg\">\n"
		, w_width, w_height
		, -w_width / 2, -w_height/ 2, w_width, w_height);
	for (int i = 0; i < GetNumTriangles(); i++)
	{
		Eigen::MatrixXf T = BGetTriangle(i);

		for (int j = 0; j < 3; j++)
			T.col(j) = View * Eigen::Vector2f(T.col(j));

		fprintf(f, "<polygon points=\"%d %d %d %d %d %d\" stroke=\"black\" fill=\"%f\" stroke-width=\"2\" />",
			int(T(0, 0) * w_width / 2), -int(T(1, 0) * w_height / 2), 
			int(T(0, 1) * w_width / 2), -int(T(1, 1) * w_height / 2), 
			int(T(0, 2) * w_width / 2), -int(T(1, 2) * w_height / 2))
		
		
	}
	fprintf(f, "</svg>\n");

	fclose(f);

	printf("[Export] Exported to %s\n", exportPathBuf);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (action == GLFW_PRESS)
		return;

    // Update the position of the first vertex if the keys 1,2, or 3 are pressed

	/*if (key == last_key)
	{
		if (key_throttle > 0)
			return;
	}
	else {
		last_key = key;
		key_throttle = 20;
	}*/

    switch (key)
    {
		case GLFW_KEY_EQUAL:
			View = Eigen::Scaling(1.2f) * View;
			break;
		case GLFW_KEY_MINUS:
			View = Eigen::Scaling(1 / 1.2f) * View;
			break;
		case GLFW_KEY_S:
			View = Eigen::Translation2f(0, -.4f) * View;
			break;
		case GLFW_KEY_W:
			View = Eigen::Translation2f(0, .4f) * View;
			break;
		case GLFW_KEY_D:
			View = Eigen::Translation2f(.4f, 0) * View;
			break;
		case GLFW_KEY_A:
			View = Eigen::Translation2f(-.4f, 0) * View;

			break;
		case GLFW_KEY_I:
			enterInsertMode();
			break;
		case GLFW_KEY_O:
			enterTranslateMode();
			break;
		case GLFW_KEY_P:
			enterDeleteMode();
			break;
		case GLFW_KEY_H:
		case GLFW_KEY_J:
			handleRotate(key);
			break;
		case GLFW_KEY_K:
		case GLFW_KEY_L:
			handleScale(key);
			break;
		case GLFW_KEY_C:
			enterColorMode();
			break;
		case GLFW_KEY_1:
		case GLFW_KEY_2:
		case GLFW_KEY_3:
		case GLFW_KEY_4:
		case GLFW_KEY_5:
		case GLFW_KEY_6:
		case GLFW_KEY_7:
		case GLFW_KEY_8:
		case GLFW_KEY_9:
			handleColorBtn(key);
			break;
		case GLFW_KEY_T:
		case GLFW_KEY_Y:
			captureKeyframe(key);
			break;
		case GLFW_KEY_U:
			toggleKeyframePlaying();
		case GLFW_KEY_M:
			exportSVG(window);
        default:
            break;
    }

    // Upload the change to the GPU
    VBO_V.update(V);
}

template<typename T>
T lerp(const T &a, const T &b, float t)
{
	return (1 - t) * a + t * b;
}

int k_targetKeyframe = 0;
float k_timer = 0, k_duration = 1, k_speed = 1;
void playKeyframe(float deltaTime)
{
	if (state != Play)
		return;

	if (V1.cols() == 0 || V2.cols() == 0)
		return;

	// Keyframe not ready
	if (V1.cols() != GetNumTriangles() * 3 || V2.cols() != GetNumTriangles() * 3)
	{
		printf("You must has the same number of triangles in two keyframes!\n");
		return;
	}

	k_timer += deltaTime;
	if (k_timer > k_duration)
	{
		k_targetKeyframe = 1 - k_targetKeyframe;
		k_timer = 0;
	}

	auto& nV = k_targetKeyframe ? V1 : V2;
	auto& nC = k_targetKeyframe ? C1 : C2;
	auto& nE = k_targetKeyframe ? E1 : E2;

	float ratio = k_timer / k_duration;
	V = lerp(V, nV, ratio);
	E = lerp(E, nE, ratio);
	C = lerp(C, nC, ratio);
}

int main(void)
{
	state = None;

    GLFWwindow* window;

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

    // Initialize the VAO
    // A Vertex Array Object (or VAO) is an object that describes how the vertex
    // attributes are stored in a Vertex Buffer Object (or VBO). This means that
    // the VAO is not the actual object storing the vertex data,
    // but the descriptor of the vertex data.
    VertexArrayObject VAO;
    VAO.init();
    VAO.bind();

    // Initialize the VBO with the vertices data
    // A VBO is a data container that lives in the GPU memory
    VBO_V.init();
	VBO_C.init();

	// Column major. (x, y)
    V << 0,  0.5, -0.5, 0.5, -0.5, -0.5;
	C << 1, 1, 1,
		 0, 0, 0,
		 0, 0, 0;
	E << 0, 0.5, 0.5, -0.5, -0.5, 0,
		 0.5, -0.5, -0.5, -0.5, -0.5, 0.5;
    VBO_V.update(V);
	VBO_C.update(C);

	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_MULTISAMPLE);
	glLineWidth(2);

    // Initialize the OpenGL Program
    // A program controls the OpenGL pipeline and it must contains
    // at least a vertex shader and a fragment shader to be valid
    Program program;
    const GLchar* vertex_shader =
            "#version 150 core\n"
                    "in vec2 position;"
					"in vec3 color;"
					"uniform mat3 view;"
					"out vec3 aColor;"
                    "void main()"
                    "{"
                    "    gl_Position = vec4((view * vec3(position, 1)).xy, 0.0, 1.0);"
					"    aColor = color;"
                    "}";
    const GLchar* fragment_shader =
            "#version 150 core\n"
					"in vec3 aColor;"
                    "out vec4 outColor;"
                    "uniform vec3 triangleColor;"
					"uniform bool useUniformColor;"
                    "void main()"
                    "{"
                    "    outColor = useUniformColor ? vec4(triangleColor, 1.0) : vec4(aColor, 1);"
                    "}";

    // Compile the two shaders and upload the binary to the GPU
    // Note that we have to explicitly specify that the output "slot" called outColor
    // is the one that we want in the fragment buffer (and thus on screen)
    program.init(vertex_shader,fragment_shader,"outColor");
    program.bind();

    // The vertex shader wants the position of the vertices as an input.
    // The following line connects the VBO we defined above with the position "slot"
    // in the vertex shader
    program.bindVertexAttribArray("position",VBO_V);
	program.bindVertexAttribArray("color", VBO_C);

    // Save the current time --- it will be used to dynamically change the triangle color
    auto t_start = std::chrono::high_resolution_clock::now();

    // Register the keyboard callback
    glfwSetKeyCallback(window, key_callback);

    // Register the mouse callback
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    // Update viewport
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	// Register the mouse move callback
	glfwSetCursorPosCallback(window, mouse_move_callback);

	auto timer = std::chrono::system_clock::now();
	float deltaTime = 0;
    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window))
    {
		playKeyframe(deltaTime);

        // Bind your VAO (not necessary if you have only one)
        VAO.bind();

        // Bind your program
        program.bind();

        // Set the uniform value depending on the time difference
        auto t_now = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration_cast<std::chrono::duration<float>>(t_now - t_start).count();
        glUniform3f(program.uniform("triangleColor"), (float)(sin(time * 4.0f) + 1.0f) / 2.0f, 0.0f, 0.0f);
		// OpenGL is column-major, and so is Eigen
		glUniformMatrix3fv(program.uniform("view"), 1, GL_FALSE, View.data());
		glUniform1i(program.uniform("useUniformColor"), 0);

        // Clear the framebuffer
        glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw a triangle
		VBO_V.update(V);
		VBO_C.update(C);
        glDrawArrays(GL_TRIANGLES, 0, V.cols());

		glUniform1i(program.uniform("useUniformColor"), 1);
		glUniform3f(program.uniform("triangleColor"), 0.f, 0.0f, 0.0f);
		VBO_V.update(E);
		glDrawArrays(GL_LINES, 0, E.cols());

        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();

		auto ttimer = std::chrono::system_clock::now();
		deltaTime = std::chrono::duration<float>(ttimer - timer).count();
		timer = ttimer;
    }

    // Deallocate opengl memory
    program.free();
    VAO.free();
    VBO_V.free();

    // Deallocate glfw internals
    glfwTerminate();
    return 0;
}
