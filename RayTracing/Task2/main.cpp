//Ray Tracing project Task 2 Anti Aliasing
//This project is implemented using the glm library, 'Ray Tracing in One Weekend' and 'Ray Tracing: The Next Week'
//Deniz Cangı
#include <iostream>
#include <fstream>
#include <glm/glm.hpp>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>

#include <memory>

using std::shared_ptr; //retains shared ownership of an object through a pointer, several shared_ptr object may have the same object
//it manages two entites: control block and obhject being managed
//pointer to some allocated type, with reference-counting semantics. Every time you assign its value to another shared pointer (usually with a simple assignment), the reference count is incremented.
//you can use auto
using std::make_shared; //performs a single heap-allocation accounting for the space necessary for both the control block and the data
using std::sqrt;
using std::pow;
const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

//inline function: to reduce the function call overhead. Inline function is a function that is expanded in line when it is called. When the inline function is called whole code of the inline function gets inserted or substituted at the point of inline function call.
inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

inline double random_double() {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

inline double random_double(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_double();
}
//clamp value x to the range min and max

inline double clamp(double x, double min, double max) {
    if (x < min)
        return min;
    if (x > max)
        return max;
    return x;
}

class Ray{
    public:
    
        glm::dvec3 origin, direction;
    
    public:
        
        Ray(const glm::dvec3 & origin, const glm::dvec3 & direction): origin(origin), direction(direction){}
        
    //    Parametric definition of the ray
    
        
        glm::dvec3 at(double t) const{ //we want to get some points on the ray
            return origin + t * direction; //start from the porigin and move t amount of direction
        }
        
        glm::dvec3 get_origin() const{
            return origin;
        }
        glm::dvec3 get_direction() const{
            return direction;
        }
};

class Camera{
    public:
        Camera(){
            auto aspect_ratio = 4.0/3.0;
            auto viewport_height = 2.0;
            auto viewport_width = aspect_ratio * viewport_height;
            auto focal_length = 1.0;

            origin = glm::dvec3(0, 0, 0);
            horizontal = glm::dvec3(viewport_width, 0, 0);
            vertical = glm::dvec3(0, viewport_height, 0);
            lower_left_corner = origin - horizontal/2. - vertical/2. - glm::dvec3(0, 0, focal_length);
        }
        
        Ray get_ray(double u, double v){
            return Ray(origin, lower_left_corner + u*horizontal + v*vertical -origin);
        }
    public:
        glm::dvec3 origin;
        glm::dvec3 lower_left_corner;
        glm::dvec3 horizontal;
        glm::dvec3 vertical;
};

struct hit_record {
    glm::dvec3 p;
    glm::dvec3 normal;
    double t;
    bool front_face;

    inline void set_face_normal(const Ray& r, const glm::dvec3 & outward_normal) {
//        if front_face is false, then dot product >0 and this means ray and normal face in the same direction, the ray is inside to object
//        if front face is true, then the dot product is >0 and ray and moral face are in the opposite direction and ray is outside the object
        front_face = glm::dot(r.get_direction(), outward_normal) < 0;
        normal = front_face ? outward_normal :-outward_normal;
    }
};

class hittable { //an abstract class that a ray might hit, to make i more convenient to use several spheres
    public:
        virtual bool hit(const Ray& r, double t_min, double t_max, hit_record& rec) const = 0; //function to see şf the ray counts using t_min and t_max.
};

//class ot list all the hittable objects:

class hittable_list : public hittable {
    public:
        hittable_list() {}
        hittable_list(shared_ptr<hittable> object) { add(object); }

        void clear() { objects.clear(); }
        void add(shared_ptr<hittable> object) { objects.push_back(object); }

        virtual bool hit(
            const Ray& r, double t_min, double t_max, hit_record& rec) const override;

    public:
        std::vector<shared_ptr<hittable>> objects;
};

bool hittable_list::hit(const Ray& r, double t_min, double t_max, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;

    for (const auto& object : objects) {
        if (object->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

class Sphere: public hittable{
    
    public:
        Sphere() {}
        Sphere(glm::dvec3 origin, double r) : origin(origin), radius(r) {};

        virtual bool hit(
            const Ray& r, double t_min, double t_max, hit_record& rec) const override;

    public:
        glm::dvec3 origin;
        double radius;
    
};

bool Sphere::hit(const Ray& r, double t_min, double t_max, hit_record& rec) const {
    glm::dvec3 oc = r.get_origin() - origin;
    auto a = glm::length(r.get_direction())*glm::length(r.get_direction());
    auto half_b = glm::dot(oc, r.get_direction());
    auto c = pow(glm::length(oc),2) - radius*radius;

    auto discriminant = half_b*half_b - a*c;
    if (discriminant < 0)
        return false;
    auto sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    glm::dvec3 outward_normal = (rec.p - origin) / radius;
    rec.set_face_normal(r, outward_normal);

    return true;
}

class xy_rect : public hittable {
    public:
        xy_rect() {}

        xy_rect(double _x0, double _x1, double _y0, double _y1, double _k)
            : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k) {};

        virtual bool hit(const Ray& r, double t_min, double t_max, hit_record& rec) const override;


    public:
        double x0, x1, y0, y1, k;
};

bool xy_rect::hit(const Ray& r, double t_min, double t_max, hit_record& rec) const {
    auto t = (k-r.get_origin()[2]) / r.get_direction()[2];
    if (t < t_min || t > t_max)
        return false;
    auto x = r.get_origin()[0] + t*r.get_direction()[0];
    auto y = r.get_origin()[1] + t*r.get_direction()[1];
    if (x < x0 || x > x1 || y < y0 || y > y1)
        return false;
    rec.t = t;
    auto outward_normal = glm::dvec3(0, 0, 1);
    rec.set_face_normal(r, outward_normal);
    rec.p = r.at(t);
    return true;
}

class xz_rect : public hittable {
    public:
        xz_rect() {}

        xz_rect(double _x0, double _x1, double _z0, double _z1, double _k)
            : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k) {};

        virtual bool hit(const Ray& r, double t_min, double t_max, hit_record& rec) const override;

    public:
        double x0, x1, z0, z1, k;
};

class yz_rect : public hittable {
    public:
        yz_rect() {}

        yz_rect(double _y0, double _y1, double _z0, double _z1, double _k)
            : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k){};

        virtual bool hit(const Ray& r, double t_min, double t_max, hit_record& rec) const override;


    public:
        double y0, y1, z0, z1, k;
};

bool xz_rect::hit(const Ray& r, double t_min, double t_max, hit_record& rec) const {
    auto t = (k-r.get_origin()[1]) / r.get_direction()[1];
    if (t < t_min || t > t_max)
        return false;
    auto x = r.get_origin()[0] + t*r.get_direction()[0];
    auto z = r.get_origin()[2] + t*r.get_direction()[2];
    if (x < x0 || x > x1 || z < z0 || z > z1)
        return false;
    rec.t = t;
    auto outward_normal = glm::dvec3(0, 1, 0);
    rec.set_face_normal(r, outward_normal);
    rec.p = r.at(t);
    return true;
}

bool yz_rect::hit(const Ray& r, double t_min, double t_max, hit_record& rec) const {
    auto t = (k-r.get_origin()[0]) / r.get_direction()[0];
    if (t < t_min || t > t_max)
        return false;
    auto y = r.get_origin()[1] + t*r.get_direction()[1];
    auto z = r.get_origin()[2] + t*r.get_direction()[2];
    if (y < y0 || y > y1 || z < z0 || z > z1)
        return false;
    rec.t = t;
    auto outward_normal = glm::dvec3(1, 0, 0);
    rec.set_face_normal(r, outward_normal);
    rec.p = r.at(t);
    return true;
}

class box : public hittable  {
    public:
        box() {}
        box(const glm::dvec3& p0, const glm::dvec3& p1);

        virtual bool hit(const Ray& r, double t_min, double t_max, hit_record& rec) const override;

    public:
        glm::dvec3 box_min;
        glm::dvec3 box_max;
        hittable_list sides;
};

box::box(const glm::dvec3& p0, const glm::dvec3& p1) {
    box_min = p0;
    box_max = p1;

    sides.add(make_shared<xy_rect>(p0[0], p1[0], p0[1], p1[1], p1[2]));
    sides.add(make_shared<xy_rect>(p0[0], p1[0], p0[1], p1[1], p0[2]));

    sides.add(make_shared<xz_rect>(p0[0], p1[0], p0[2], p1[2], p1[1]));
    sides.add(make_shared<xz_rect>(p0[0], p1[0], p0[2], p1[2], p0[1]));

    sides.add(make_shared<yz_rect>(p0[1], p1[1], p0[2], p1[2], p1[0]));
    sides.add(make_shared<yz_rect>(p0[1], p1[1], p0[2], p1[2], p0[0]));
}

bool box::hit(const Ray& r, double t_min, double t_max, hit_record& rec) const {
    return sides.hit(r, t_min, t_max, rec);
}

glm::dvec3 color_of_ray(const Ray& r, const hittable& world){
    hit_record rec;
    
    if (world.hit(r, 0, infinity, rec)) {
        return 0.5 * (rec.normal + glm::dvec3(1,1,1));
    }
    glm::dvec3 unit_direction = glm::normalize(r.get_direction());
    auto t = 0.5*(unit_direction[1] + 1.0);
    return (1.0-t)*glm::dvec3(1.0, 1.0, 1.0) + t*glm::dvec3(0.5, 0.7, 1.0);
}

void render(const int & x_dimension, const int & y_dimension, Camera camera, hittable_list world){
    
    std::ofstream output;
    const int samples_per_pixel = 200;
    output.open("./project20.ppm", std::ios::out | std::ios::trunc);
    output << "P3\n" << x_dimension << ' ' << y_dimension << "\n255\n";
    
    for (int j = y_dimension-1; j >= 0; --j) {
        for (int i = 0; i < x_dimension; ++i) {
            glm::dvec3 color(0,0,0);
            for (int s=0; s < samples_per_pixel; ++s){
                auto u = (double(i) + random_double()) / (x_dimension-1);
                auto v = (double(j) + random_double()) / (y_dimension-1);
                Ray ray = camera.get_ray(u, v);
                glm::dvec3 x = color_of_ray(ray,world);
                color += x;
            }
            
            auto r = color[0];
            auto g = color[1];
            auto b = color[2];

            // Divide the color by the number of samples and gamma-correct for gamma=2.0.
            auto scale = 1.0 / samples_per_pixel;
            r = scale * r;
            g = scale * g;
            b = scale * b;
            
            output << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
            << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
            << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';
            
        }
    }
    output.close();
    std::cout << "done!" << std::endl;
}

int main() {

    // Image

    const int x_dimension = 640;
    const int y_dimension = 480;
    
    hittable_list world;
//    Spheres to add
    world.add(make_shared<Sphere>(glm::dvec3(0,-100.5,-1), 100));
    world.add(make_shared<Sphere>(glm::dvec3(0, 0,-1), 0.25));
    world.add(make_shared<Sphere>(glm::dvec3(0.8,-0.2,-2), 0.2));
    world.add(make_shared<Sphere>(glm::dvec3(1.3,-0.25,-2), 0.1));
    world.add(make_shared<Sphere>(glm::dvec3(-0.8,-0.2,-2), 0.2));
    world.add(make_shared<Sphere>(glm::dvec3(-1.3,-0.25,-2), 0.1));
    
//    big sphere at the bottom
    
    Camera camera;
    // Render
    
    render(x_dimension, y_dimension,camera,world);
}
