#ifndef __SHADER_H_
#define __SHADER_H_

#include <Eigen/Geometry>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <string>
#include <cstring>
#include <cstdio>

struct Path {
	std::string path;
	Path(std::string path) : path(path) {}
};

class Shader
{
public:
	unsigned id;
	Shader();
	Shader(const char *vshader, const char *fshader);
	~Shader();

	bool loaded();
	void use();
	static Shader& now();

	void setUnif(const std::string &name, bool value) const;
	void setUnif(const std::string &name, int value) const;
	void setUnif(const std::string &name, unsigned value) const;
	void setUnif(const std::string &name, float value) const;
	void setUnif(const std::string &name, glm::mat2 &mat) const;
	void setUnif(const std::string &name, glm::mat3 &mat) const;
	void setUnif(const std::string &name, glm::mat4 &mat) const;
	void setUnif(const std::string& name, Eigen::Affine3f trans) const;
	void setUnif(const std::string &name, glm::vec2 &vec) const;
	void setUnif(const std::string &name, glm::vec3 &vec) const;
	void setUnif(const std::string &name, glm::vec4 &vec) const;

private:
	static Shader* current;

};
#endif