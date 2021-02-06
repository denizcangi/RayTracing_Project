//Ray Tracing project Task1 Basic Scene
//This project is implemented using the glm library, 'Ray Tracing in One Weekend' and 'Ray Tracing: The Next Week'
//Deniz Cangı
#include <iostream>
#include <fstream>
#include <glm/glm.hpp>
#include <vector>
#include <cmath>
#include <cstdlib>

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

            origin = glm::dvec3(0, 1, 1);
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
    output.open("./task1.ppm", std::ios::out | std::ios::trunc);
    output << "P3\n" << x_dimension << ' ' << y_dimension << "\n255\n";
    
    for (int j = y_dimension-1; j >= 0; --j) {
        for (int i = 0; i < x_dimension; ++i) {
            double u;
            double v;
            u = double(i) / (x_dimension-1);
            v = double(j) / (y_dimension-1);
            Ray r = camera.get_ray(u, v);
            glm::dvec3 color = color_of_ray(r,world);
            
            auto x = static_cast<int>(255.999 * color[0]);
            auto y = static_cast<int>(255.999 * color[1]);
            auto z = static_cast<int>(255.999 * color[2]);
            
            output << x << ' ' << y << ' ' << z << '\n';
            
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
    world.add(make_shared<Sphere>(glm::dvec3(0,0,-1), 0.25));
    world.add(make_shared<Sphere>(glm::dvec3(0.8,-0.2,-2), 0.2));
    world.add(make_shared<Sphere>(glm::dvec3(1.3,-0.25,-2), 0.1));
    world.add(make_shared<Sphere>(glm::dvec3(-0.8,-0.2,-2), 0.2));
    world.add(make_shared<Sphere>(glm::dvec3(-1.3,-0.25,-2), 0.1));
    
//    big sphere at the bottom
    
    world.add(make_shared<Sphere>(glm::dvec3(0,-100.5,-1), 100));
    
    Camera camera;
    // Render
    
    render(x_dimension, y_dimension,camera,world);
}
