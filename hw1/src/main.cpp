// C++ include
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/task_group.h>

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"
#include "utils.h"

// Shortcut to avoid Eigen:: and std:: everywhere, DO NOT USE IN .h
using namespace std;
using namespace Eigen;

#define HIT_NONE            0
#define HIT_FRONT           1
#define HIT_BACK            2
#define HIT_FRONT_AND_BACK  (HIT_FRONT|HIT_BACK)
#define BIGFLOAT 1.0e30f
#define EPS 0.001


class Node;
class Light;

class Ray
{
public:
    Vector3d p;
    Vector3d dir;
 
    Ray() {}
    Ray(const Vector3d &_p, const Vector3d &_dir) : p(_p), dir(_dir) {}
    Ray(const Ray &r) : p(r.p), dir(r.dir) {}
    void Normalize() { dir.normalize(); }
};

struct HitInfo
{
    float z;            // the distance from the ray center to the hit point
    Vector3d p;           // position of the hit point
    Vector3d N;           // surface normal at the hit point
    std::shared_ptr<Node> node;   // the object node that was hit
    bool front;         // true if the ray hits the front side, false if the ray hits the back side
 
    HitInfo() { Init(); }
    void Init() { z=BIGFLOAT; node=nullptr; front=true; }
};

class Object
{
public:
	virtual bool IntersectRay( const Ray &ray, HitInfo &hInfo, int hitSide=HIT_FRONT ) const = 0;
};

class Material
{
public:
    virtual Vector3d Shade(const Ray &ray, const HitInfo &hInfo, const std::vector<std::shared_ptr<Light>> &lights, const std::vector<std::shared_ptr<Node>> nodes,  int bounceCount) const=0;
};

class Node
{
private:
    std::shared_ptr<Object> obj;
    std::shared_ptr<Material> mtl;
public:
    Node(): obj(nullptr), mtl(nullptr) {}
    Node(const std::shared_ptr<Object>& o, const std::shared_ptr<Material>& m): obj(o), mtl(m) {}
    const std::shared_ptr<Object>   GetNodeObj() const { return obj; }
    std::shared_ptr<Object>         GetNodeObj() { return obj; }
    void            SetNodeObj(std::shared_ptr<Object> object) { obj=object; }
 
    // Material management
    const std::shared_ptr<Material> GetMaterial() const { return mtl; }
    void            SetMaterial(std::shared_ptr<Material> material) { mtl=material; }

};

void trace(const Ray& ray, std::vector<std::shared_ptr<Node>> nodes, HitInfo& hInfo, bool hitBack) {
    for (const auto& node : nodes) {
        const std::shared_ptr<Object> obj = node->GetNodeObj();

        // Ray local_ray = node->ToNodeCoords(ray);
        bool res = false;

        if (obj) {
            float z = hInfo.z;
            obj->IntersectRay(ray, hInfo, hitBack ? HIT_FRONT_AND_BACK : HIT_FRONT);
            if (hInfo.z < z) {
                hInfo.node = node;
            }
        }
    }
}


class Light
{
public:
    Light(): ambient(false) {}
    virtual Vector3d   Illuminate(const Vector3d &p, const Vector3d &N, std::vector<std::shared_ptr<Node>> nodes) const=0;
    virtual Vector3d  Direction (const Vector3d &p) const=0;
    virtual bool    IsAmbient () const { return ambient; }
    virtual void setAmbient(bool am) {ambient = am;}
    static float Shadow(Ray ray, std::vector<std::shared_ptr<Node>> nodes, float t_max=BIGFLOAT) {
        // return 1.0;
        HitInfo info;                
        trace(ray, nodes, info, true);
        if (info.node != nullptr && info.z < t_max) {
            return 0.0f;
        }
        else {
            return 1;
        }
    }
private:
    bool ambient;
};

class Sphere : public Object
{
public:
    Vector3d center;
    double radius;
    Sphere(Vector3d center, double radius){
        this->center = center;
        this->radius = radius;
    }
	bool IntersectRay( const Ray &ray, HitInfo &hInfo, int hitSide=HIT_FRONT ) const override;
};

bool Sphere::IntersectRay(const Ray& ray, HitInfo& hInfo, int hitSide) const {
    float a = ray.dir.dot(ray.dir);
    float b = ray.dir.dot(ray.p - center);
    float c = (ray.p - center).dot(ray.p - center) - radius * radius;
    float det = b * b - a * c;
    if (det < 0) {
        return false;
    }
    else {
        float t1 = (-b - sqrt(det)) / a;
        float t2 = (-b + sqrt(det)) / a;
        if (t1 > t2) {
            std::swap(t1, t2);
        }
        float tmin = min(t1, t2);
        float tmax = max(t1, t2);

        if (hitSide != HIT_FRONT) {
			if (hInfo.z > tmax && tmax > 0.001) {
				hInfo.z = tmax;
				hInfo.p = ray.p + tmax * ray.dir;
				hInfo.N = hInfo.p;
                hInfo.front = false;
			}
		}
		if (hitSide != HIT_NONE) {
			if (hInfo.z > tmin && tmin > 0.001) {
				hInfo.z = tmin;
				hInfo.p = ray.p + tmin * ray.dir;
				hInfo.N = hInfo.p;
                hInfo.front = true;
			}
		}
		return true;
    }
}

class PointLight : public Light
{
public:
    PointLight(const Vector3d& i, const Vector3d& p) : Light(), intensity(i), position(p) {}
    virtual Vector3d Illuminate(const Vector3d &p, const Vector3d &N, std::vector<std::shared_ptr<Node>> nodes) const { return Shadow(Ray(p,position-p), nodes, 100000) * intensity; }
    virtual Vector3d Direction(const Vector3d &p) const { return (p-position).normalized(); }
    void SetIntensity(const Vector3d& intens) { intensity=intens; }
    void SetPosition(const Vector3d& pos) { position=pos; }
 
private:
    Vector3d intensity;
    Vector3d position;
};

class AmbientLight : public Light
{
public:
    AmbientLight(const Vector3d& i) : intensity(i) {}
    virtual Vector3d Illuminate(const Vector3d &p, const Vector3d &N, std::vector<std::shared_ptr<Node>> nodes) const { return intensity; }
    virtual Vector3d Direction(const Vector3d &p) const { return Vector3d(0,0,0); }
    virtual bool IsAmbient() const { return true; }
 
    void SetIntensity(Vector3d intens) { intensity=intens; }
private:
    Vector3d intensity;
};
 
//-------------------------------------------------------------------------------
 
class DirectLight : public Light
{
public:
    DirectLight(const Vector3d& i, const Vector3d& d, const bool& s) : intensity(i), direction(d), shadow(s) {}
    virtual Vector3d Illuminate(const Vector3d &p, const Vector3d &N, std::vector<std::shared_ptr<Node>> nodes) const { 
        if (shadow) {
            return Shadow(Ray(p,-direction), nodes) * intensity;
        }
        else {
            return intensity; 
        }
    }
    virtual Vector3d Direction(const Vector3d &p) const { return direction; }
 
    void SetIntensity(const Vector3d& intens) { intensity=intens; }
    void SetDirection(const Vector3d& dir) { direction=dir.normalized(); }
private:
    Vector3d intensity;
    Vector3d direction;
    bool shadow;
};


class Diffuse : public Material
{
public:
    Diffuse(const Vector3d& d) : diffuse(d) {}
    Vector3d Shade(const Ray &ray, const HitInfo &hInfo, const std::vector<std::shared_ptr<Light>> &lights, const std::vector<std::shared_ptr<Node>> nodes, int bounceCount) const override {
        Vector3d res(0, 0, 0);
        for (const auto& light : lights) {
            Vector3d lightColor = light->Illuminate(hInfo.p, hInfo.N, nodes);
            if (light->IsAmbient()) {
                res += lightColor.cwiseProduct(diffuse);
            }
            else {
                Vector3d L = -light->Direction(hInfo.p);
                Vector3d R = -ray.dir;
                if (L.dot(hInfo.N) > 0 && R.dot(hInfo.N) > 0) {
                    float lamb = L.dot(hInfo.N);
                    Vector3d lightColor = light->Illuminate(hInfo.p, hInfo.N, nodes);
                    res += lamb * diffuse.cwiseProduct(lightColor);
                }
            }
        }
        res[0] = min(max(res[0], 0.0), 1.0);
        res[1] = min(max(res[1], 0.0), 1.0);
        res[2] = min(max(res[2], 0.0), 1.0);
        return res;
    }
 
    void SetDiffuse(Vector3d dif) { diffuse = dif; }
 
private:
    Vector3d diffuse;
    float glossiness;
};

class PhongShading : public Material
{
public:
    PhongShading(const Vector3d& d, const Vector3d& s, const float& g) : diffuse(d), specular(s), glossiness(g) {}
    virtual Vector3d Shade(const Ray &ray, const HitInfo &hInfo, const std::vector<std::shared_ptr<Light>> &lights, const std::vector<std::shared_ptr<Node>> nodes, int bounceCount) const {
        Vector3d res(0, 0, 0);
        for (const auto& light : lights) {
            Vector3d lightColor = light->Illuminate(hInfo.p, hInfo.N, nodes);
            if (light->IsAmbient()) {
                res += lightColor.cwiseProduct(diffuse);
            }
            else {
                Vector3d L = -light->Direction(hInfo.p);
                Vector3d R = -ray.dir;
                Vector3d H = (L + R).normalized();
                float s = pow(max(H.dot(hInfo.N), 0.0), glossiness);
                float lamb = max(L.dot(hInfo.N), 0.0);
                res += lamb * diffuse.cwiseProduct(lightColor) + lamb * s * specular.cwiseProduct(lightColor);
            }
        }
        res[0] = min(max(res[0], 0.0), 1.0);
        res[1] = min(max(res[1], 0.0), 1.0);
        res[2] = min(max(res[2], 0.0), 1.0);
        return res;
    }
 
    void SetDiffuse(Vector3d dif) { diffuse = dif; }
    void SetSpecular(Vector3d spec) { specular = spec; }
    void SetGlossiness(float gloss) { glossiness = gloss; }
 
private:
    Vector3d diffuse, specular;
    float glossiness;
};

float fresnel_reflection(const Ray& ray, const Vector3d& N, const float& ior, bool front) {
    float cos_i = abs((-ray.dir).dot(N));
    float sin_i = sqrt(1 - cos_i * cos_i);
    float ior_i = 1.0f;
    float ior_t = ior;
    if (!front) {
        std::swap(ior_i, ior_t);
    }
    float sin_t = ior_i * sin_i / ior_t;
    if (sin_t >= 1.0f) {
        return 1.0f;
    }
    else {
        float cos_t = sqrt(1 - sin_t * sin_t);
        float r1 = (ior_t * cos_i - ior_i * cos_t) / (ior_t * cos_i + ior_i * cos_t);
        float r2 = (ior_i * cos_i - ior_t * cos_t) / (ior_i * cos_i + ior_t * cos_t);
        return 0.5 * (r1 * r1 + r2 * r2);
    }
}


Ray reflect(const Ray& ray, const HitInfo &hInfo) {
    Ray reflect_ray;
    reflect_ray.p = hInfo.p;
    reflect_ray.dir = ray.dir - 2.0f * (hInfo.N.dot(ray.dir)) * hInfo.N;
    reflect_ray.Normalize();
    return reflect_ray;
}

class MtlBlinn : public Material
{
public:
    MtlBlinn(const Vector3d& d, const Vector3d& s, const float& g, const Vector3d& r) : diffuse(d), specular(s), glossiness(g), reflection(r), ior(1) {}
    virtual Vector3d Shade(const Ray &ray, const HitInfo &hInfo, const std::vector<std::shared_ptr<Light>> &lights, const std::vector<std::shared_ptr<Node>> nodes, int bounceCount) const {
        Vector3d res(0, 0, 0);
        float kr = fresnel_reflection(ray, hInfo.N, ior, hInfo.front);
        if (bounceCount > 0) {
            // reflection
            Ray reflect_ray = reflect(ray, hInfo);
            HitInfo reflect_info;
            trace(reflect_ray, nodes, reflect_info, HIT_FRONT_AND_BACK);
            if (reflect_info.node != nullptr) {
                res += reflect_info.node->GetMaterial()->Shade(reflect_ray, reflect_info, lights, nodes, bounceCount - 1).cwiseProduct(Vector3d(reflection[0]+kr, reflection[1]+kr, reflection[2]+kr));
            }
        }

        // material
        for (const auto& light : lights) {
            Vector3d lightColor = light->Illuminate(hInfo.p, hInfo.N, nodes);
            if (light->IsAmbient()) {
                res += lightColor.cwiseProduct(diffuse);
            }
            else {
                Vector3d L = -light->Direction(hInfo.p);
                Vector3d R = -ray.dir;
                Vector3d H = (L + R).normalized();
                float s = pow(max(H.dot(hInfo.N), 0.0), glossiness);
                float lamb = max(L.dot(hInfo.N), 0.0);
                res += lamb * diffuse.cwiseProduct(lightColor) + lamb * s * specular.cwiseProduct(lightColor);
            }
        }
        res[0] = min(max(res[0], 0.0), 1.0);
        res[1] = min(max(res[1], 0.0), 1.0);
        res[2] = min(max(res[2], 0.0), 1.0);
        return res;
    }
 
    void SetDiffuse(Vector3d dif) { diffuse = dif; }
    void SetSpecular(Vector3d spec) { specular = spec; }
    void SetGlossiness(float gloss) { glossiness = gloss; }
 
    void SetReflection(Vector3d reflect) { reflection = reflect; }
    void SetRefractionIndex(float _ior) { ior = _ior; }
 
private:
    Vector3d diffuse, specular, reflection;
    float glossiness;
    float ior;  // index of refraction
};

Vector3d cross(const Vector3d& a, const Vector3d& b) {
    Vector3d c;
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
    return c;
}


class Mesh : public Object {
public:
    Mesh(std::string filename) {
        std::ifstream infile(filename);
        if (infile.bad()) {
            std::cerr << "Unable to open " << filename << "\n";
            exit(-1);
        }
        std::string line;
        std::getline(infile, line);
        std::getline(infile, line);
        int nn;
        stringstream ss1(line);
        ss1 >> nv >> nf >> nn;
        verts.resize(nv);
        vnorms.resize(nv);
        faces.resize(nf);
        fnorms.resize(nf);

        for (int i = 0; i < nv; ++i) {
            std::getline(infile, line);
            stringstream ss(line);
            float x, y, z;
            ss >> x >> y >> z;
            verts[i] = Vector3d(x, y, z);
        }
        for (int i = 0; i < nv; ++i) {
            std::getline(infile, line);
            stringstream ss(line);
            int n, i1, i2, i3;
            ss >> n >> i1 >> i2 >> i3;
            faces[i] = Vector3i(i1, i2, i3);
        }

        infile.close();

        for (int i = 0; i < nv; ++i) {
            vnorms[i] = Vector3d(0, 0, 0);
        }
        for (int i = 0; i < nf; ++i) {
            Vector3d N = cross(verts[faces[i][1]] - verts[faces[i][0]], verts[faces[i][2]] - verts[faces[i][0]]);
            vnorms[faces[i][0]] += N;
            vnorms[faces[i][1]] += N;
            vnorms[faces[i][2]] += N;
            fnorms[i] = N;
        }
        for (int i = 0; i < nv; ++i) {
            vnorms[i] = vnorms[i].normalized();
        }

        cout << "Finish reading " << filename << ", " << nv << " vertices and " << nf << " faces loaded\n";
    }
    virtual bool IntersectRay( const Ray &ray, HitInfo &hInfo, int hitSide=HIT_FRONT ) const;

private:
    std::vector<Vector3d> verts;
    std::vector<Vector3i> faces;
    std::vector<Vector3d> fnorms;
    std::vector<Vector3d> vnorms;
    int nv, nf;
    bool IntersectTriangle( const Ray &ray, HitInfo &hInfo, int hitSide, unsigned int faceID ) const;
};

bool Mesh::IntersectTriangle( const Ray &ray, HitInfo &hInfo, int hitSide, unsigned int faceID ) const {
    Vector3i face = faces[faceID];
    Vector3d e1 = verts[face[1]] - verts[face[0]];
    Vector3d e2 = verts[face[2]] - verts[face[0]];
    Vector3d p = cross(ray.dir, e2);
    float a = e1.dot(p);
    if (a == 0.0) {
        return false;
    }
    /*if (hitSide == HIT_FRONT && a < 0) {
        return false;
    }
    if (hitSide == HIT_BACK && a > 0) {
        return false;
    }*/
    float f = 1.0 / a;
    Vector3d s = ray.p - verts[face[0]];
    float u = f * (s.dot(p));
    if (u < 0.0 || u > 1.0) {
        return false;
    }
    Vector3d q = cross(s, e1);
    float v = f * (ray.dir.dot(q));
    if (v < 0.0 || u + v > 1.0) {
        return false;
    }
    float t = f * (e2.dot(q));
    Vector3d bc(1.0 - (u + v), u, v);

    if (hInfo.z > t && t > EPS) {
        hInfo.N = vnorms[faces[faceID][0]] * bc[0] + vnorms[faces[faceID][1]] * bc[1] + vnorms[faces[faceID][2]] * bc[2];
        hInfo.N = hInfo.N.normalized();
        hInfo.p = ray.p + t * ray.dir;
        hInfo.z = t;
        hInfo.front = (a > 0.0f);
        return true;
    }

    return false;
}

bool Mesh::IntersectRay( const Ray &ray, HitInfo &hInfo, int hitSide) const {
    bool res = false;
    for (int fid = 0; fid < nf; ++fid) {
        res |= IntersectTriangle(ray, hInfo, hitSide, fid);
    }
    return res;
}

class Plane : public Object
{
public:
    Plane(float y): height(y) {}
    virtual bool IntersectRay( const Ray &ray, HitInfo &hInfo, int hitSide=HIT_FRONT ) const;
private:
    float height;
};

bool Plane::IntersectRay( const Ray &ray, HitInfo &hInfo, int hitSide) const {
    Vector3d N(0, 1, 0);
    float det = ray.dir.dot(N);
    if (det < 0 && hitSide == HIT_BACK) {
        return false;
    }
    if (det >= 0 && hitSide == HIT_FRONT) {
        return false;
    }
    float t = -(ray.p[1] - height) / ray.dir[1];
    Vector3d p = ray.p + t * ray.dir;
    //if (abs(p[0]) > 1.0f || abs(p.y) > 1.0f) {
    //    return false;
    //}

    if (t > EPS && hInfo.z > t) {
        hInfo.front = (det < 0);
        hInfo.N = N;
        hInfo.p = p;
        hInfo.z = t;
        return true;
    }
    else {
        return false;
    }
}

void part1()
{
    std::cout << "Part 1: Simple ray tracer, four spheres with orthographic projection" << std::endl;

    const std::string filename("part1.png");
    MatrixXd C = MatrixXd::Zero(800,800); // Store the color C
    MatrixXd R = MatrixXd::Zero(800,800); // Store the color R
    MatrixXd G = MatrixXd::Zero(800,800); // Store the color G 
    MatrixXd B = MatrixXd::Zero(800,800); // Store the color B
    MatrixXd A = MatrixXd::Zero(800,800); // Store the alpha mask

    // The camera is orthographic, pointing in the direction -z and covering the unit square (-1,1) in x and y
    Vector3d origin(-1,1,0.85);
    Vector3d x_displacement(2.0/C.cols(),0,0);
    Vector3d y_displacement(0,-2.0/C.rows(),0);

    std::vector<std::shared_ptr<Light>> lights(3);
    lights[0] = std::make_shared<AmbientLight>(Vector3d(0.2,0.2,0.2));
    lights[1] = std::make_shared<DirectLight>(Vector3d(0.5,0.4,0.4), Vector3d(-1,-1,-1), false);
    lights[2] = std::make_shared<DirectLight>(Vector3d(0.2,0.7,0.7), Vector3d(1,1,-1), false);
    // lights[3] = std::make_shared<PointLight>(Vector3d(20.7,20.7,20.7), Vector3d(0.45,0.45,-0.3));
    // lights[1]->setAmbient(true);

    // Prepare 4 spheres
    double sphere_radius = 0.3;

    std::shared_ptr<Object> sph1 = std::make_shared<Sphere>(Vector3d( 0.4, 0.4, -0.0), 0.4);
    std::shared_ptr<Object> sph2 = std::make_shared<Sphere>(Vector3d( 0.4,-0.4, 0.1), 0.3);
    std::shared_ptr<Object> sph3 = std::make_shared<Sphere>(Vector3d(-0.4, 0.4, 0.0), 0.3);
    std::shared_ptr<Object> sph4 = std::make_shared<Sphere>(Vector3d(-0.4,-0.4, 0.1), 0.3);

    std::vector<std::shared_ptr<Node>> nodes(4);
    nodes[0] = std::make_shared<Node>(sph1, std::make_shared<Diffuse>(Vector3d(0.7, 0.3, 0.3)));
    nodes[1] = std::make_shared<Node>(sph2, std::make_shared<Diffuse>(Vector3d(0.3, 0.7, 0.3)));
    nodes[2] = std::make_shared<Node>(sph3, std::make_shared<Diffuse>(Vector3d(0.3, 0.3, 0.7)));
    nodes[3] = std::make_shared<Node>(sph4, std::make_shared<Diffuse>(Vector3d(0.7, 0.7, 0.3)));

    for (unsigned i=0;i<C.cols();i++)
    {
        for (unsigned j=0;j<C.rows();j++)
        {
            // Prepare the ray
            Vector3d ray_origin = origin + double(i)*x_displacement + double(j)*y_displacement;
            Vector3d ray_direction = Vector3d(0,0,-1);
            Ray r(ray_origin, ray_direction);
            r.Normalize();
            HitInfo info;
            trace(r, nodes, info, false);
            Vector3d color;

            if (info.node) {
                color = info.node->GetMaterial()->Shade(r, info, lights, nodes, 5);
            }
            else {
                color = Vector3d(0, 0, 0);
            }
            
            R(i, j) = color[0];
            G(i, j) = color[1];
            B(i, j) = color[2];

            A(i,j) = 1;
        }
    }
    // Save to png
    write_matrix_to_png(R,G,B,A,filename);
}

void part2()
{
    std::cout << "Part 2: Simple ray tracer, four spheres with orthographic projection, different material properties" << std::endl;

    const std::string filename("part2.png");
    MatrixXd C = MatrixXd::Zero(800,800); // Store the color C
    MatrixXd R = MatrixXd::Zero(800,800); // Store the color R
    MatrixXd G = MatrixXd::Zero(800,800); // Store the color G 
    MatrixXd B = MatrixXd::Zero(800,800); // Store the color B
    MatrixXd A = MatrixXd::Zero(800,800); // Store the alpha mask

    // The camera is orthographic, pointing in the direction -z and covering the unit square (-1,1) in x and y
    Vector3d origin(-1,1,0.85);
    Vector3d x_displacement(2.0/C.cols(),0,0);
    Vector3d y_displacement(0,-2.0/C.rows(),0);

    // Single light source
    std::vector<std::shared_ptr<Light>> lights(4);
    lights[0] = std::make_shared<AmbientLight>(Vector3d(0.1,0.1,0.1));
    lights[1] = std::make_shared<DirectLight>(Vector3d(0.5,0.4,0.4), Vector3d(-1,-1,-1), false);
    lights[2] = std::make_shared<DirectLight>(Vector3d(0.2,0.7,0.7), Vector3d(1,1,-1), false);
    lights[3] = std::make_shared<DirectLight>(Vector3d(1.7,1.7,1.7), Vector3d(1,-1,-1), false);

    // Prepare 4 spheres
    std::shared_ptr<Object> sph1 = std::make_shared<Sphere>(Vector3d( 0.4, 0.4, -0.0), 0.4);
    std::shared_ptr<Object> sph2 = std::make_shared<Sphere>(Vector3d( 0.4,-0.4, 0.1), 0.3);
    std::shared_ptr<Object> sph3 = std::make_shared<Sphere>(Vector3d(-0.4, 0.4, 0.0), 0.3);
    std::shared_ptr<Object> sph4 = std::make_shared<Sphere>(Vector3d(-0.4,-0.4, 0.1), 0.3);

    std::vector<std::shared_ptr<Node>> nodes(4);
    nodes[0] = std::make_shared<Node>(sph1, std::make_shared<Diffuse>(Vector3d(0.7, 0.3, 0.3)));
    nodes[1] = std::make_shared<Node>(sph2, std::make_shared<PhongShading>(Vector3d(0.2, 0.4, 0.2), Vector3d(1.3, 0.7, 0.3), 200));
    nodes[2] = std::make_shared<Node>(sph3, std::make_shared<PhongShading>(Vector3d(0.2, 0.2, 0.4), Vector3d(0.3, 0.3, 0.7), 210));
    nodes[3] = std::make_shared<Node>(sph4, std::make_shared<PhongShading>(Vector3d(0.4, 0.4, 0.2), Vector3d(0.7, 0.7, 0.3), 220));

    for (unsigned i=0;i<C.cols();i++)
    {
        for (unsigned j=0;j<C.rows();j++)
        {
            // Prepare the ray
            Vector3d ray_origin = origin + double(i)*x_displacement + double(j)*y_displacement;
            Vector3d ray_direction = Vector3d(0,0,-1);
            Ray r(ray_origin, ray_direction);
            r.Normalize();
            HitInfo info;
            trace(r, nodes, info, false);
            Vector3d color;

            if (info.node) {
                color = info.node->GetMaterial()->Shade(r, info, lights, nodes, 2);
            }
            else {
                color = Vector3d(0, 0, 0);
            }
            
            R(i, j) = color[0];
            G(i, j) = color[1];
            B(i, j) = color[2];

            A(i,j) = 1;
        }
    }
    // Save to png
    write_matrix_to_png(R,G,B,A,filename);
}


void part3()
{
    std::cout << "Part 3: Simple ray tracer, four spheres with perspective projection, different material properties" << std::endl;

    const std::string filename("part3.png");
    MatrixXd C = MatrixXd::Zero(800,800); // Store the color C
    MatrixXd R = MatrixXd::Zero(800,800); // Store the color R
    MatrixXd G = MatrixXd::Zero(800,800); // Store the color G 
    MatrixXd B = MatrixXd::Zero(800,800); // Store the color B
    MatrixXd A = MatrixXd::Zero(800,800); // Store the alpha mask

    // The camera is orthographic, pointing in the direction -z and covering the unit square (-1,1) in x and y
    int w = C.cols();
    int h = C.rows();
    Vector3d cameraPos = Vector3d(0,0,1.3);
    Vector3d cameraUp = Vector3d(0,1,0);
    Vector3d cameraDir = Vector3d(0,0,-1);
    float fov = 90;
    Vector3d left = cross(cameraUp, cameraDir);
    left = left.normalized();
    float plane_h = tanf(fov * 0.5 * M_PI / 180.0f) * 2.0f;
    float plane_w = plane_h / h * w;
    Vector3d base = cameraPos + cameraDir + plane_w * 0.5 * left + plane_h * 0.5 * cameraUp;
    Vector3d dx = -plane_w / w * left;
    Vector3d dy = -plane_h / h * cameraUp;

    // Single light source
    std::vector<std::shared_ptr<Light>> lights(4);
    lights[0] = std::make_shared<AmbientLight>(Vector3d(0.1,0.1,0.1));
    lights[1] = std::make_shared<DirectLight>(Vector3d(0.5,0.4,0.4), Vector3d(-1,-1,-1), false);
    lights[2] = std::make_shared<DirectLight>(Vector3d(0.2,0.7,0.7), Vector3d(1,1,-1), false);
    lights[3] = std::make_shared<DirectLight>(Vector3d(1.7,1.7,1.7), Vector3d(1,-1,-1), false);

    // Prepare 4 spheres
    std::shared_ptr<Object> sph1 = std::make_shared<Sphere>(Vector3d( 0.4, 0.4, 0.1), 0.4);
    std::shared_ptr<Object> sph2 = std::make_shared<Sphere>(Vector3d( 0.4,-0.4, 0.1), 0.3);
    std::shared_ptr<Object> sph3 = std::make_shared<Sphere>(Vector3d(-0.4, 0.4, 0.0), 0.3);
    std::shared_ptr<Object> sph4 = std::make_shared<Sphere>(Vector3d(-0.4,-0.4, 0.1), 0.3);

    std::vector<std::shared_ptr<Node>> nodes(4);
    nodes[0] = std::make_shared<Node>(sph1, std::make_shared<PhongShading>(Vector3d(0.7, 0.3, 0.3), Vector3d(0, 0, 0), 200));
    nodes[1] = std::make_shared<Node>(sph2, std::make_shared<PhongShading>(Vector3d(0.2, 0.4, 0.2), Vector3d(1.3, 0.7, 0.3), 200));
    nodes[2] = std::make_shared<Node>(sph3, std::make_shared<PhongShading>(Vector3d(0.2, 0.2, 0.4), Vector3d(0.3, 0.3, 0.7), 210));
    nodes[3] = std::make_shared<Node>(sph4, std::make_shared<PhongShading>(Vector3d(0.4, 0.4, 0.2), Vector3d(0.7, 0.7, 0.3), 220));

    for (unsigned i=0;i<C.cols();i++)
    {
        for (unsigned j=0;j<C.rows();j++)
        {
            Ray r;
            HitInfo info;
            r.p = cameraPos;
            r.dir = base + (0.5f + i) * dx + (0.5f + j) * dy - cameraPos;
            r.Normalize();
            trace(r, nodes, info, false);
            Vector3d color;

            if (info.node) {
                color = info.node->GetMaterial()->Shade(r, info, lights, nodes, 2);
            }
            else {
                color = Vector3d(0, 0, 0);
            }
            
            R(i, j) = color[0];
            G(i, j) = color[1];
            B(i, j) = color[2];

            A(i,j) = 1;
        }
    }
    // Save to png
    write_matrix_to_png(R,G,B,A,filename);
}

void part4()
{
    std::cout << "Part 4: Simple ray tracer, render loaded off objects" << std::endl;

    int width = 800;
    int height = 800;

    const std::string filename("part4.png");
    MatrixXd C = MatrixXd::Zero(height,width); // Store the color C
    MatrixXd R = MatrixXd::Zero(height,width); // Store the color R
    MatrixXd G = MatrixXd::Zero(height,width); // Store the color G 
    MatrixXd B = MatrixXd::Zero(height,width); // Store the color B
    MatrixXd A = MatrixXd::Zero(height,width); // Store the alpha mask

    // The camera is orthographic, pointing in the direction -z and covering the unit square (-1,1) in x and y
    int w = C.cols();
    int h = C.rows();
    Vector3d cameraPos = Vector3d(0,0.1,0.2);
    Vector3d cameraUp = Vector3d(0,1,0);
    Vector3d cameraDir = Vector3d(0,0,-1);
    float fov = 90;
    Vector3d left = cross(cameraUp, cameraDir);
    left = left.normalized();
    float plane_h = tanf(fov * 0.5 * M_PI / 180.0f) * 2.0f;
    float plane_w = plane_h / h * w;
    Vector3d base = cameraPos + cameraDir + plane_w * 0.5 * left + plane_h * 0.5 * cameraUp;
    Vector3d dx = -plane_w / w * left;
    Vector3d dy = -plane_h / h * cameraUp;

    // Single light source
    std::vector<std::shared_ptr<Light>> lights(4);
    lights[0] = std::make_shared<AmbientLight>(Vector3d(0.1,0.1,0.1));
    lights[1] = std::make_shared<DirectLight>(Vector3d(0.5,0.4,0.4), Vector3d(-1,-1,-1), false);
    lights[2] = std::make_shared<PointLight>(Vector3d(2.2,2.7,2.7), Vector3d(0,0.3,0));
    lights[3] = std::make_shared<DirectLight>(Vector3d(1.7,1.7,1.7), Vector3d(1,-1,-1), false);

    // Prepare 4 spheres
    // std::shared_ptr<Object> off1 = std::make_shared<Mesh>("data/bumpy_cube.off");
    std::shared_ptr<Object> off2 = std::make_shared<Mesh>("data/bunny.off");

    std::vector<std::shared_ptr<Node>> nodes(1);
    nodes[0] = std::make_shared<Node>(off2, std::make_shared<PhongShading>(Vector3d(0.7, 0.3, 0.3), Vector3d(0, 0, 0), 200));
    // nodes[1] = std::make_shared<Node>(off2, std::make_shared<PhongShading>(Vector3d(0.2, 0.4, 0.2), Vector3d(1.3, 0.7, 0.3), 200));

    auto render_column = [&](int i) {
        for (unsigned j=0;j<C.rows();j++)
        {
            Ray r;
            HitInfo info;
            r.p = cameraPos;
            r.dir = base + (0.5f + i) * dx + (0.5f + j) * dy - cameraPos;
            r.Normalize();
            trace(r, nodes, info, false);
            Vector3d color;

            if (info.node) {
                color = info.node->GetMaterial()->Shade(r, info, lights, nodes, 5);
            }
            else {
                color = Vector3d(0, 0, 0);
            }
            
            R(i, j) = color[0];
            G(i, j) = color[1];
            B(i, j) = color[2];

            A(i,j) = 1;
        }
    };

    tbb::parallel_for(tbb::blocked_range<int>(0, C.cols()), [&render_column](const tbb::blocked_range<int>& range) {
        for (int i = range.begin(); i < range.end(); ++i) {
            render_column(i);
        }
    });
    // Save to png
    write_matrix_to_png(R,G,B,A,filename);
}

void part5()
{
    std::cout << "Part 5: Simple ray tracer, shadow supported" << std::endl;

    int width = 800;
    int height = 800;

    const std::string filename("part5.png");
    MatrixXd C = MatrixXd::Zero(height,width); // Store the color C
    MatrixXd R = MatrixXd::Zero(height,width); // Store the color R
    MatrixXd G = MatrixXd::Zero(height,width); // Store the color G 
    MatrixXd B = MatrixXd::Zero(height,width); // Store the color B
    MatrixXd A = MatrixXd::Zero(height,width); // Store the alpha mask

    // The camera is orthographic, pointing in the direction -z and covering the unit square (-1,1) in x and y
    int w = C.cols();
    int h = C.rows();
    Vector3d cameraPos = Vector3d(0,0.1,0.25);
    Vector3d cameraUp = Vector3d(0,1,0);
    Vector3d cameraDir = Vector3d(0,0,-1);
    float fov = 90;
    Vector3d left = cross(cameraUp, cameraDir);
    left = left.normalized();
    float plane_h = tanf(fov * 0.5 * M_PI / 180.0f) * 2.0f;
    float plane_w = plane_h / h * w;
    Vector3d base = cameraPos + cameraDir + plane_w * 0.5 * left + plane_h * 0.5 * cameraUp;
    Vector3d dx = -plane_w / w * left;
    Vector3d dy = -plane_h / h * cameraUp;

    // Single light source
    std::vector<std::shared_ptr<Light>> lights(4);
    lights[0] = std::make_shared<AmbientLight>(Vector3d(0.1,0.1,0.1));
    lights[1] = std::make_shared<DirectLight>(Vector3d(0.5,0.4,0.4), Vector3d(-1,-1,-1), true);
    lights[2] = std::make_shared<PointLight>(Vector3d(2.2,2.7,2.7), Vector3d(0,0.3,0));
    lights[3] = std::make_shared<DirectLight>(Vector3d(1.7,1.7,1.7), Vector3d(1,-1,-1), true);

    // Prepare 4 spheres
    std::shared_ptr<Object> off1 = std::make_shared<Plane>(0.0);
    std::shared_ptr<Object> off2 = std::make_shared<Mesh>("data/bunny.off");

    std::vector<std::shared_ptr<Node>> nodes(2);
    nodes[0] = std::make_shared<Node>(off1, std::make_shared<PhongShading>(Vector3d(0.7, 0.3, 0.3), Vector3d(0, 0, 0), 200));
    nodes[1] = std::make_shared<Node>(off2, std::make_shared<PhongShading>(Vector3d(0.2, 0.4, 0.2), Vector3d(1.3, 0.7, 0.3), 200));

    auto render_column = [&](int i) {
        for (unsigned j=0;j<C.rows();j++)
        {
            Ray r;
            HitInfo info;
            r.p = cameraPos;
            r.dir = base + (0.5f + i) * dx + (0.5f + j) * dy - cameraPos;
            r.Normalize();
            trace(r, nodes, info, false);
            Vector3d color;

            if (info.node) {
                color = info.node->GetMaterial()->Shade(r, info, lights, nodes, 5);
            }
            else {
                color = Vector3d(0, 0, 0);
            }
            
            R(i, j) = color[0];
            G(i, j) = color[1];
            B(i, j) = color[2];

            A(i,j) = 1;
        }
    };

    tbb::parallel_for(tbb::blocked_range<int>(0, C.cols()), [&render_column](const tbb::blocked_range<int>& range) {
        for (int i = range.begin(); i < range.end(); ++i) {
            render_column(i);
        }
    });
    // Save to png
    write_matrix_to_png(R,G,B,A,filename);
}

void part6()
{
    std::cout << "Part 6: Simple ray tracer, reflection supported" << std::endl;

    int width = 800;
    int height = 800;

    const std::string filename("part6.png");
    MatrixXd C = MatrixXd::Zero(height,width); // Store the color C
    MatrixXd R = MatrixXd::Zero(height,width); // Store the color R
    MatrixXd G = MatrixXd::Zero(height,width); // Store the color G 
    MatrixXd B = MatrixXd::Zero(height,width); // Store the color B
    MatrixXd A = MatrixXd::Zero(height,width); // Store the alpha mask

    // The camera is orthographic, pointing in the direction -z and covering the unit square (-1,1) in x and y
    int w = C.cols();
    int h = C.rows();
    Vector3d cameraPos = Vector3d(0,0.1,0.25);
    Vector3d cameraUp = Vector3d(0,1,0);
    Vector3d cameraDir = Vector3d(0,0,-1);
    float fov = 90;
    Vector3d left = cross(cameraUp, cameraDir);
    left = left.normalized();
    float plane_h = tanf(fov * 0.5 * M_PI / 180.0f) * 2.0f;
    float plane_w = plane_h / h * w;
    Vector3d base = cameraPos + cameraDir + plane_w * 0.5 * left + plane_h * 0.5 * cameraUp;
    Vector3d dx = -plane_w / w * left;
    Vector3d dy = -plane_h / h * cameraUp;

    // Single light source
    std::vector<std::shared_ptr<Light>> lights(3);
    lights[0] = std::make_shared<AmbientLight>(Vector3d(0.1,0.1,0.1));
    lights[1] = std::make_shared<DirectLight>(Vector3d(0.5,0.4,0.4), Vector3d(-1,-1,-1), true);
    // lights[2] = std::make_shared<PointLight>(Vector3d(2.2,2.7,2.7), Vector3d(0,0.3,0));
    lights[2] = std::make_shared<DirectLight>(Vector3d(1.7,1.7,1.7), Vector3d(1,-1,-1), true);

    // Prepare 4 spheres
    std::shared_ptr<Object> off1 = std::make_shared<Plane>(0.0);
    std::shared_ptr<Object> off2 = std::make_shared<Mesh>("data/bunny.off");

    std::vector<std::shared_ptr<Node>> nodes(2);
    nodes[0] = std::make_shared<Node>(off1, std::make_shared<MtlBlinn>(Vector3d(0.1, 0.1, 0.1), Vector3d(0, 0, 0), 200, Vector3d(0.8, 0.8, 0.8)));
    nodes[1] = std::make_shared<Node>(off2, std::make_shared<PhongShading>(Vector3d(0.2, 0.4, 0.2), Vector3d(1.3, 0.7, 0.3), 200));

    auto render_column = [&](int i) {
        for (unsigned j=0;j<C.rows();j++)
        {
            Ray r;
            HitInfo info;
            r.p = cameraPos;
            r.dir = base + (0.5f + i) * dx + (0.5f + j) * dy - cameraPos;
            r.Normalize();
            trace(r, nodes, info, false);
            Vector3d color;

            if (info.node) {
                color = info.node->GetMaterial()->Shade(r, info, lights, nodes, 5);
            }
            else {
                color = Vector3d(0, 0, 0);
            }
            
            R(i, j) = color[0];
            G(i, j) = color[1];
            B(i, j) = color[2];

            A(i,j) = 1;
        }
    };

    tbb::parallel_for(tbb::blocked_range<int>(0, C.cols()), [&render_column](const tbb::blocked_range<int>& range) {
        for (int i = range.begin(); i < range.end(); ++i) {
            render_column(i);
        }
    });
    // Save to png
    write_matrix_to_png(R,G,B,A,filename);
}

void part7()
{
    std::cout << "Part 7: Simple ray tracer, TBB Acceleration" << std::endl;

    int width = 800;
    int height = 800;

    const std::string filename("part7.png");
    MatrixXd C = MatrixXd::Zero(height,width); // Store the color C
    MatrixXd R = MatrixXd::Zero(height,width); // Store the color R
    MatrixXd G = MatrixXd::Zero(height,width); // Store the color G 
    MatrixXd B = MatrixXd::Zero(height,width); // Store the color B
    MatrixXd A = MatrixXd::Zero(height,width); // Store the alpha mask

    // The camera is orthographic, pointing in the direction -z and covering the unit square (-1,1) in x and y
    int w = C.cols();
    int h = C.rows();
    Vector3d cameraPos = Vector3d(0,0.1,0.25);
    Vector3d cameraUp = Vector3d(0,1,0);
    Vector3d cameraDir = Vector3d(0,0,-1);
    float fov = 90;
    Vector3d left = cross(cameraUp, cameraDir);
    left = left.normalized();
    float plane_h = tanf(fov * 0.5 * M_PI / 180.0f) * 2.0f;
    float plane_w = plane_h / h * w;
    Vector3d base = cameraPos + cameraDir + plane_w * 0.5 * left + plane_h * 0.5 * cameraUp;
    Vector3d dx = -plane_w / w * left;
    Vector3d dy = -plane_h / h * cameraUp;

    // Single light source
    std::vector<std::shared_ptr<Light>> lights(3);
    lights[0] = std::make_shared<AmbientLight>(Vector3d(0.1,0.1,0.1));
    lights[1] = std::make_shared<DirectLight>(Vector3d(0.5,0.4,0.4), Vector3d(-1,-1,-1), true);
    // lights[2] = std::make_shared<PointLight>(Vector3d(2.2,2.7,2.7), Vector3d(0,0.3,0));
    lights[2] = std::make_shared<DirectLight>(Vector3d(1.7,1.7,1.7), Vector3d(1,-1,-1), true);

    // Prepare 4 spheres
    std::shared_ptr<Object> off1 = std::make_shared<Plane>(0.0);
    std::shared_ptr<Object> off2 = std::make_shared<Mesh>("data/bunny.off");

    std::vector<std::shared_ptr<Node>> nodes(2);
    nodes[0] = std::make_shared<Node>(off1, std::make_shared<MtlBlinn>(Vector3d(0.1, 0.1, 0.1), Vector3d(0, 0, 0), 200, Vector3d(0.8, 0.8, 0.8)));
    nodes[1] = std::make_shared<Node>(off2, std::make_shared<PhongShading>(Vector3d(0.2, 0.4, 0.2), Vector3d(1.3, 0.7, 0.3), 200));

    int counter = 0;

    auto render_column = [&](int i) {
        for (unsigned j=0;j<C.rows();j++)
        {
            Ray r;
            HitInfo info;
            r.p = cameraPos;
            r.dir = base + (0.5f + i) * dx + (0.5f + j) * dy - cameraPos;
            r.Normalize();
            trace(r, nodes, info, false);
            Vector3d color;

            if (info.node) {
                color = info.node->GetMaterial()->Shade(r, info, lights, nodes, 5);
            }
            else {
                color = Vector3d(0, 0, 0);
            }
            
            R(i, j) = color[0];
            G(i, j) = color[1];
            B(i, j) = color[2];

            A(i,j) = 1;
        }
    };

    tbb::parallel_for(tbb::blocked_range<int>(0, C.cols()), [&render_column](const tbb::blocked_range<int>& range) {
        for (int i = range.begin(); i < range.end(); ++i) {
            render_column(i);
        }
    });

    
    // Save to png
    write_matrix_to_png(R,G,B,A,filename);
}

void part8()
{
    std::cout << "Part 8: Simple ray tracer, Animation" << std::endl;

    int width = 400;
    int height = 400;

    MatrixXd C = MatrixXd::Zero(height,width); // Store the color C
    MatrixXd R = MatrixXd::Zero(height,width); // Store the color R
    MatrixXd G = MatrixXd::Zero(height,width); // Store the color G 
    MatrixXd B = MatrixXd::Zero(height,width); // Store the color B
    MatrixXd A = MatrixXd::Zero(height,width); // Store the alpha mask
    int w = C.cols();
    int h = C.rows();

    // Single light source
    std::vector<std::shared_ptr<Light>> lights(3);
    lights[0] = std::make_shared<AmbientLight>(Vector3d(0.1,0.1,0.1));
    lights[1] = std::make_shared<DirectLight>(Vector3d(0.5,0.4,0.4), Vector3d(-1,-1,-1), true);
    // lights[2] = std::make_shared<PointLight>(Vector3d(2.2,2.7,2.7), Vector3d(0,0.3,0));
    lights[2] = std::make_shared<DirectLight>(Vector3d(1.7,1.7,1.7), Vector3d(1,-1,-1), true);

    // Prepare 4 spheres
    std::shared_ptr<Object> off1 = std::make_shared<Plane>(0.0);
    std::shared_ptr<Object> off2 = std::make_shared<Mesh>("data/bunny.off");

    std::vector<std::shared_ptr<Node>> nodes(2);
    nodes[0] = std::make_shared<Node>(off1, std::make_shared<MtlBlinn>(Vector3d(0.1, 0.1, 0.1), Vector3d(0, 0, 0), 200, Vector3d(0.8, 0.8, 0.8)));
    nodes[1] = std::make_shared<Node>(off2, std::make_shared<PhongShading>(Vector3d(0.2, 0.4, 0.2), Vector3d(1.3, 0.7, 0.3), 200));

    for (int i = 0; i < 30; i++) {
        const std::string filename("part8_" + std::to_string(i) + ".png");

        Vector3d cameraPos = Vector3d(0.25*sin(i * M_PI/6), 0.1, 0.25*cos(i * M_PI/6));
        Vector3d cameraUp = Vector3d(0,1,0);
        Vector3d cameraDir = Vector3d(0, 0, 0) - cameraPos;
        float fov = 90;
        Vector3d left = cross(cameraUp, cameraDir);
        left = left.normalized();
        float plane_h = tanf(fov * 0.5 * M_PI / 180.0f) * 2.0f;
        float plane_w = plane_h / h * w;
        Vector3d base = cameraPos + cameraDir + plane_w * 0.5 * left + plane_h * 0.5 * cameraUp;
        Vector3d dx = -plane_w / w * left;
        Vector3d dy = -plane_h / h * cameraUp;


        int counter = 0;

        auto render_column = [&](int i) {
            for (unsigned j=0;j<C.rows();j++)
            {
                Ray r;
                HitInfo info;
                r.p = cameraPos;
                r.dir = base + (0.5f + i) * dx + (0.5f + j) * dy - cameraPos;
                r.Normalize();
                trace(r, nodes, info, false);
                Vector3d color;

                if (info.node) {
                    color = info.node->GetMaterial()->Shade(r, info, lights, nodes, 5);
                }
                else {
                    color = Vector3d(0, 0, 0);
                }
                
                R(i, j) = color[0];
                G(i, j) = color[1];
                B(i, j) = color[2];

                A(i,j) = 1;
            }
        };

        tbb::parallel_for(tbb::blocked_range<int>(0, C.cols()), [&render_column](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); ++i) {
                render_column(i);
            }
        });

        
        // Save to png
        write_matrix_to_png(R,G,B,A,filename);
    }
}

int main()
{
    // part1();
    // part2();
    // part3();
    // part4();
    part5();
    part6();
    // part7();
    // part8();
    return 0;
}