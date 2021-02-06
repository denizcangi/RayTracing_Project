//Ray Tracing project Task 3 More Shapes
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

inline static glm::dvec3 random_withoutparameters() {
    return glm::dvec3(random_double(), random_double(), random_double());
}

inline static glm::dvec3 random(double min, double max) {
    return glm::dvec3(random_double(min,max), random_double(min,max), random_double(min,max));
}

glm::dvec3 random_in_unit_sphere() {
    while (true) {
        auto p = random(-1,1);
        if (pow(glm::length(p),2) >= 1)
            continue;
        return p;
    }
}

glm::dvec3 random_unit_vector() {
    return glm::normalize(random_in_unit_sphere());
}

bool near_zero(glm::dvec3 x){
    // Return true if the vector is close to zero in all dimensions.
    const auto s = 1e-8;
    return (fabs(x[0]) < s) && (fabs(x[1]) < s) && (fabs(x[2]) < s);
}

inline int random_int(int min, int max) {
    // Returns a random integer in [min,max].
    return static_cast<int>(random_double(min, max+1));
}

glm::dvec3 random_in_unit_disk() {
    while (true) {
        auto p = glm::dvec3(random_double(-1,1), random_double(-1,1), 0);
        if (pow(glm::length(p),2) >= 1) continue;
        return p;
    }
}

class Ray{
    public:
    
        glm::dvec3 origin, direction;
        double tm;
    
    public:
    
        Ray(){}
        
        Ray(const glm::dvec3 & origin, const glm::dvec3 & direction,double time = 0.0): origin(origin), direction(direction), tm(time){}
        
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
        
        double get_time() const{
            return tm;
        }
};

class Camera {
    public:
        Camera(
            glm::dvec3 lookfrom,
            glm::dvec3 lookat,
            glm::dvec3 vup,
            double vfov, // vertical field-of-view in degrees
            double aspect_ratio,
            double aperture,
            double focus_dist,
            double _time0 = 0,
            double _time1 = 0
        ) {
            auto theta = degrees_to_radians(vfov);
            auto h = tan(theta/2);
            auto viewport_height = 2.0 * h;
            auto viewport_width = aspect_ratio * viewport_height;

            w = glm::normalize(lookfrom - lookat);
            u = glm::normalize(glm::cross(vup, w));
            v = glm::cross(w, u);

            origin = lookfrom;
            horizontal = focus_dist * viewport_width * u;
            vertical = focus_dist * viewport_height * v;
            lower_left_corner = origin - horizontal/2. - vertical/2. - focus_dist*w;

            lens_radius = aperture / 2.;
            time0 = _time0;
            time1 = _time1;
        }

        Ray get_ray(double s, double t) const {
            glm::dvec3 rd = lens_radius * random_in_unit_disk();
            glm::dvec3  offset = u * rd[0] + v * rd[1];

            return Ray(origin + offset,lower_left_corner + s*horizontal + t*vertical - origin - offset, random_double(time0, time1));
        }

    private:
        glm::dvec3 origin;
        glm::dvec3 lower_left_corner;
        glm::dvec3 horizontal;
        glm::dvec3 vertical;
        glm::dvec3 u, v, w;
        double lens_radius;
        double time0, time1;  // shutter open/close times
};

class material;

struct hit_record {
    glm::dvec3 p;
    glm::dvec3 normal;
    shared_ptr<material> mat_ptr;
    double t;
    double u;
    double v;
    bool front_face;

    inline void set_face_normal(const Ray& r, const glm::dvec3 & outward_normal) {
//        if front_face is false, then dot product >0 and this means ray and normal face in the same direction, the ray is inside to object
//        if front face is true, then the dot product is >0 and ray and moral face are in the opposite direction and ray is outside the object
        front_face = glm::dot(r.get_direction(), outward_normal) < 0;
        normal = front_face ? outward_normal :-outward_normal;
    }
};
class aabb {
    public:
        aabb() {}
        aabb(const glm::dvec3& a, const glm::dvec3& b) { minimum = a; maximum = b;}

        glm::dvec3 min() const {return minimum; }
        glm::dvec3 max() const {return maximum; }

        inline bool hit(const Ray& r, double t_min, double t_max) const {
            for (int a = 0; a < 3; a++) {
                auto invD = 1.0f / r.get_direction()[a];
                auto t0 = (min()[a] - r.get_origin()[a]) * invD;
                auto t1 = (max()[a] - r.get_origin()[a]) * invD;
                if (invD < 0.0f)
                    std::swap(t0, t1);
                t_min = t0 > t_min ? t0 : t_min;
                t_max = t1 < t_max ? t1 : t_max;
                if (t_max <= t_min)
                    return false;
            }
            return true;
        }

        glm::dvec3 minimum;
        glm::dvec3 maximum;
};

class hittable { //an abstract class that a ray might hit, to make i more convenient to use several spheres
    public:
        virtual bool hit(const Ray& r, double t_min, double t_max, hit_record& rec) const = 0; //function to see şf the ray counts using t_min and t_max.
        virtual bool bounding_box(double time0, double time1, aabb& output_box) const = 0;
};



class material {
    public:
        virtual glm::dvec3 emitted(double u, double v, const glm::dvec3 & p) const {
            return glm::dvec3 (0,0,0);
        }
    
        virtual bool scatter(
            const Ray& r_in, const hit_record& rec, glm::dvec3& attenuation, Ray& scattered
        ) const = 0;
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
        virtual bool bounding_box(double time0, double time1, aabb& output_box) const override;

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

aabb surrounding_box(aabb box0, aabb box1) {
    glm::dvec3 small(fmin(box0.min()[0], box1.min()[0]),
                 fmin(box0.min()[1], box1.min()[1]),
                     fmin(box0.min()[2], box1.min()[2]));

    glm::dvec3 big(fmax(box0.max()[0], box1.max()[0]),
               fmax(box0.max()[1], box1.max()[1]),
               fmax(box0.max()[2], box1.max()[2]));

    return aabb(small,big);
}

bool hittable_list::bounding_box(double time0, double time1, aabb& output_box) const {
    if (objects.empty()) return false;

    aabb temp_box;
    bool first_box = true;

    for (const auto& object : objects) {
        if (!object->bounding_box(time0, time1, temp_box)) return false;
        output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
        first_box = false;
    }

    return true;
}

class Sphere: public hittable{
    
    public:
        Sphere() {}
        Sphere(glm::dvec3 origin, double r, shared_ptr<material> m) : origin(origin), radius(r), mat_ptr(m) {};

        virtual bool hit(
            const Ray& r, double t_min, double t_max, hit_record& rec) const override;
        virtual bool bounding_box(double time0, double time1, aabb& output_box) const override;

    public:
        glm::dvec3 origin;
        double radius;
        shared_ptr<material> mat_ptr;
    private:
        static void get_sphere_uv(const glm::dvec3& p, double& u, double& v) {
            // p: a given point on the sphere of radius one, centered at the origin.
            // u: returned value [0,1] of angle around the Y axis from X=-1.
            // v: returned value [0,1] of angle from Y=-1 to Y=+1.
            //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
            //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
            //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

            auto theta = acos(-p[1]);
            auto phi = atan2(-p[2], p[1]) + pi;

            u = phi / (2*pi);
            v = theta / pi;
        }
    
    
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
    get_sphere_uv(outward_normal, rec.u, rec.v);
    rec.mat_ptr = mat_ptr;

    return true;
}

bool Sphere::bounding_box(double time0, double time1, aabb& output_box) const {
    output_box = aabb(origin - glm::dvec3(radius, radius, radius), origin + glm::dvec3(radius, radius, radius));
    return true;
}


//Texture class

class texture {
    public:
        virtual glm::dvec3 value(double u, double v, const glm::dvec3& p) const = 0;
};

class solid_color : public texture {
    public:
        solid_color() {}
        solid_color(glm::dvec3 c) : color_value(c) {}

        solid_color(double red, double green, double blue)
          : solid_color(glm::dvec3(red,green,blue)) {}

        virtual glm::dvec3 value(double u, double v, const glm::dvec3& p) const {
            return color_value;
        }

    private:
        glm::dvec3 color_value;
};

//Lambertian material class
//it can either scatter always and attenuate by its reflectance 𝑅, or it can scatter with no attenuation but absorb the fraction 1−R of the rays, or it could be a mixture of those strategies.

class lambertian : public material {
    public:
        lambertian(glm::dvec3 a) : albedo(make_shared<solid_color>(a)) {}
        lambertian(shared_ptr<texture> a) : albedo(a) {}

        virtual bool scatter(
            const Ray& r_in, const hit_record& rec, glm::dvec3& attenuation, Ray& scattered
        ) const override {
            auto scatter_direction = rec.normal + random_unit_vector();
            if (near_zero(scatter_direction))
                scatter_direction = rec.normal;
            
            scattered = Ray(rec.p, scatter_direction, r_in.get_time());
            attenuation = albedo->value(rec.u, rec.v, rec.p);
            return true;
        }

    public:
        shared_ptr<texture> albedo;
};

glm::dvec3 reflect(const glm::dvec3& v, const glm::dvec3& n) {
    return v - 2*glm::dot(v,n)*n;
}

class metal : public material {
    public:
        metal(const glm::dvec3& a) : albedo(a) {}

        virtual bool scatter(
            const Ray& r_in, const hit_record& rec, glm::dvec3& attenuation, Ray& scattered
        ) const override {
            glm::dvec3 reflected = reflect(glm::normalize(r_in.get_direction()), rec.normal);
            scattered = Ray(rec.p, reflected,r_in.get_time());
            attenuation = albedo;
            return (glm::dot(scattered.get_direction(), rec.normal) > 0);
        }

    public:
        glm::dvec3 albedo;
};

glm::dvec3 refract(const glm::dvec3& uv, const glm::dvec3 & n, double etai_over_etat) {
    auto cos_theta = fmin(glm::dot(-uv, n), 1.0);
    glm::dvec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    glm::dvec3 r_out_parallel = -sqrt(fabs(1.0 - pow(glm::length(r_out_perp),2))) * n;
    return r_out_perp + r_out_parallel;
}

class dielectric : public material {
    public:
        dielectric(double index_of_refraction) : ir(index_of_refraction) {}

        virtual bool scatter(
            const Ray& r_in, const hit_record& rec, glm::dvec3& attenuation, Ray& scattered
        ) const override {
            attenuation = glm::dvec3(1.0, 1.0, 1.0);
            double refraction_ratio = rec.front_face ? (1.0/ir) : ir;

            glm::dvec3 unit_direction = glm::normalize(r_in.get_direction());
            
            double cos_theta = fmin(glm::dot(-unit_direction, rec.normal), 1.0);
            double sin_theta = sqrt(1.0 - cos_theta*cos_theta);

            bool cannot_refract = refraction_ratio * sin_theta > 1.0;
            glm::dvec3 direction;

            if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double())
                direction = reflect(unit_direction, rec.normal);
            else
                direction = refract(unit_direction, rec.normal, refraction_ratio);

            scattered = Ray(rec.p, direction,r_in.get_time());
            
            glm::dvec3 refracted = refract(unit_direction, rec.normal, refraction_ratio);

            scattered = Ray(rec.p, refracted);
            return true;
        }

    public:
        double ir; // Index of Refraction
    private:
        static double reflectance(double cosine, double ref_idx) {
            // Use Schlick's approximation for reflectance.
            auto r0 = (1-ref_idx) / (1+ref_idx);
            r0 = r0*r0;
            return r0 + (1-r0)*pow((1 - cosine),5);
        }
};

class checker_texture : public texture {
    public:
        checker_texture() {}

        checker_texture(shared_ptr<texture> _even, shared_ptr<texture> _odd)
            : even(_even), odd(_odd) {}

        checker_texture(glm::dvec3 c1, glm::dvec3 c2)
            : even(make_shared<solid_color>(c1)) , odd(make_shared<solid_color>(c2)) {}

        virtual glm::dvec3 value(double u, double v, const glm::dvec3& p) const override {
            auto sines = sin(10*p[0])*sin(10*p[1])*sin(10*p[2]);
            if (sines < 0)
                return odd->value(u, v, p);
            else
                return even->value(u, v, p);
        }

    public:
        shared_ptr<texture> odd;
        shared_ptr<texture> even;
};

class perlin {
    public:
        perlin() {
            ranfloat = new double[point_count];
            for (int i = 0; i < point_count; ++i) {
                ranfloat[i] = random_double();
            }

            perm_x = perlin_generate_perm();
            perm_y = perlin_generate_perm();
            perm_z = perlin_generate_perm();
        }

        ~perlin() {
            delete[] ranfloat;
            delete[] perm_x;
            delete[] perm_y;
            delete[] perm_z;
        }

        double noise(const glm::dvec3& p) const {
            auto i = static_cast<int>(4*p[0]) & 255;
            auto j = static_cast<int>(4*p[1]) & 255;
            auto k = static_cast<int>(4*p[2]) & 255;

            return ranfloat[perm_x[i] ^ perm_y[j] ^ perm_z[k]];
        }

    private:
        static const int point_count = 256;
        double* ranfloat;
        int* perm_x;
        int* perm_y;
        int* perm_z;

        static int* perlin_generate_perm() {
            auto p = new int[point_count];

            for (int i = 0; i < perlin::point_count; i++)
                p[i] = i;

            permute(p, point_count);

            return p;
        }

        static void permute(int* p, int n) {
            for (int i = n-1; i > 0; i--) {
                int target = random_int(0, i);
                int tmp = p[i];
                p[i] = p[target];
                p[target] = tmp;
            }
        }
};

class noise_texture : public texture {
    public:
        noise_texture() {}
        noise_texture(double sc) : scale(sc) {}

        virtual glm::dvec3 value(double u, double v, const glm::dvec3 & p) const override {
            return glm::dvec3(1,1,1) * noise.noise(p);
        }

    public:
        perlin noise;
        double scale;
};

class diffuse_light : public material  {
    public:
        diffuse_light(shared_ptr<texture> a) : emit(a) {}
        diffuse_light(glm::dvec3 c) : emit(make_shared<solid_color>(c)) {}

        virtual bool scatter(
            const Ray& r_in, const hit_record& rec, glm::dvec3& attenuation, Ray& scattered
        ) const override {
            return false;
        }

        virtual glm::dvec3 emitted(double u, double v, const glm::dvec3& p) const override{
            return emit->value(u, v, p);
        }

    public:
        shared_ptr<texture> emit;
};

class xy_rect : public hittable {
    public:
        xy_rect() {}

        xy_rect(double _x0, double _x1, double _y0, double _y1, double _k,
            shared_ptr<material> mat)
            : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(mat) {};

        virtual bool hit(const Ray& r, double t_min, double t_max, hit_record& rec) const override;

        virtual bool bounding_box(double time0, double time1, aabb& output_box) const override {
            // The bounding box must have non-zero width in each dimension, so pad the Z
            // dimension a small amount.
            output_box = aabb(glm::dvec3(x0,y0, k-0.0001), glm::dvec3(x1, y1, k+0.0001));
            return true;
        }

    public:
        shared_ptr<material> mp;
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
    rec.u = (x-x0)/(x1-x0);
    rec.v = (y-y0)/(y1-y0);
    rec.t = t;
    auto outward_normal = glm::dvec3(0, 0, 1);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mp;
    rec.p = r.at(t);
    return true;
}

class xz_rect : public hittable {
    public:
        xz_rect() {}

        xz_rect(double _x0, double _x1, double _z0, double _z1, double _k,
            shared_ptr<material> mat)
            : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mp(mat) {};

        virtual bool hit(const Ray& r, double t_min, double t_max, hit_record& rec) const override;

        virtual bool bounding_box(double time0, double time1, aabb& output_box) const override {
            // The bounding box must have non-zero width in each dimension, so pad the Y
            // dimension a small amount.
            output_box = aabb(glm::dvec3(x0,k-0.0001,z0), glm::dvec3(x1, k+0.0001, z1));
            return true;
        }

    public:
        shared_ptr<material> mp;
        double x0, x1, z0, z1, k;
};

class yz_rect : public hittable {
    public:
        yz_rect() {}

        yz_rect(double _y0, double _y1, double _z0, double _z1, double _k,
            shared_ptr<material> mat)
            : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mp(mat) {};

        virtual bool hit(const Ray& r, double t_min, double t_max, hit_record& rec) const override;

        virtual bool bounding_box(double time0, double time1, aabb& output_box) const override {
            // The bounding box must have non-zero width in each dimension, so pad the X
            // dimension a small amount.
            output_box = aabb(glm::dvec3(k-0.0001, y0, z0), glm::dvec3(k+0.0001, y1, z1));
            return true;
        }

    public:
        shared_ptr<material> mp;
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
    rec.u = (x-x0)/(x1-x0);
    rec.v = (z-z0)/(z1-z0);
    rec.t = t;
    auto outward_normal = glm::dvec3(0, 1, 0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mp;
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
    rec.u = (y-y0)/(y1-y0);
    rec.v = (z-z0)/(z1-z0);
    rec.t = t;
    auto outward_normal = glm::dvec3(1, 0, 0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mp;
    rec.p = r.at(t);
    return true;
}

class box : public hittable  {
    public:
        box() {}
        box(const glm::dvec3& p0, const glm::dvec3& p1, shared_ptr<material> ptr);

        virtual bool hit(const Ray& r, double t_min, double t_max, hit_record& rec) const override;

        virtual bool bounding_box(double time0, double time1, aabb& output_box) const override {
            output_box = aabb(box_min, box_max);
            return true;
        }

    public:
        glm::dvec3 box_min;
        glm::dvec3 box_max;
        hittable_list sides;
};

box::box(const glm::dvec3& p0, const glm::dvec3& p1, shared_ptr<material> ptr) {
    box_min = p0;
    box_max = p1;

    sides.add(make_shared<xy_rect>(p0[0], p1[0], p0[1], p1[1], p1[2], ptr));
    sides.add(make_shared<xy_rect>(p0[0], p1[0], p0[1], p1[1], p0[2], ptr));

    sides.add(make_shared<xz_rect>(p0[0], p1[0], p0[2], p1[2], p1[1], ptr));
    sides.add(make_shared<xz_rect>(p0[0], p1[0], p0[2], p1[2], p0[1], ptr));

    sides.add(make_shared<yz_rect>(p0[1], p1[1], p0[2], p1[2], p1[0], ptr));
    sides.add(make_shared<yz_rect>(p0[1], p1[1], p0[2], p1[2], p0[0], ptr));
}

bool box::hit(const Ray& r, double t_min, double t_max, hit_record& rec) const {
    return sides.hit(r, t_min, t_max, rec);
}

glm::dvec3 color_of_ray(const Ray& r, const hittable& world, int depth, const glm::dvec3& background){
    hit_record rec;
    if (depth <= 0)
        return glm::dvec3(0,0,0);
    
    // If the ray hits nothing, return the background color.
    if (!world.hit(r, 0.001, infinity, rec))
        return background;

    Ray scattered;
    glm::dvec3 attenuation;
    glm::dvec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

    if (!rec.mat_ptr->scatter(r, rec, attenuation, scattered))
        return emitted;

    return emitted + attenuation * color_of_ray(scattered, world, depth-1,background);
}

hittable_list task3(){
    
    hittable_list world;
    auto material_ground = make_shared<lambertian>(glm::dvec3(0.1, 0.2, 0.5));
    auto material_left   = make_shared<dielectric>(1.5);
    auto material_left2   = make_shared<dielectric>(15);
    auto material_center = make_shared<lambertian>(glm::dvec3(0.7, 0.3, 0.3));
    
    auto pertext = make_shared<noise_texture>(4.0);
    
    world.add(make_shared<Sphere>(glm::dvec3( 0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(make_shared<Sphere>(glm::dvec3( 0.0,    -0.08, -1.0),   0.25, material_center));
    world.add(make_shared<Sphere>(glm::dvec3(-0.45,    -0.085, -1.0),   0.2, material_center));
    world.add(make_shared<Sphere>(glm::dvec3( 0.45,    -0.085, -1.0),   0.2, material_center));
    world.add(make_shared<Sphere>(glm::dvec3(-0.85,    -0.115, -1.0),   0.15, material_center));
    world.add(make_shared<Sphere>(glm::dvec3( 0.85,    -0.115, -1.0),   0.15, material_center));
    world.add(make_shared<box>(glm::dvec3(1,-0.5,0), glm::dvec3(1.5,0,0.5),material_center));
    world.add(make_shared<box>(glm::dvec3(-1.5,-0.5,0), glm::dvec3(-1,0.5,0.5),material_center));
    auto difflight = make_shared<diffuse_light>(glm::dvec3(0.5,0.5,0.5));
    world.add(make_shared<Sphere>(glm::dvec3(0,110,0),100,difflight));
    
    return world;
}

void render(const int & x_dimension, const int & y_dimension, Camera camera, hittable_list world){
    
    std::ofstream output;
    const int samples_per_pixel = 100;
    const int max_depth = 50;
    glm::dvec3 background(0,0,0);
    output.open("./task3.ppm", std::ios::out | std::ios::trunc);
    output << "P3\n" << x_dimension << ' ' << y_dimension << "\n255\n";
    
    for (int j = y_dimension-1; j >= 0; --j) {
        for (int i = 0; i < x_dimension; ++i) {
            glm::dvec3 color(0,0,0);
            for (int s=0; s < samples_per_pixel; ++s){
                auto u = (double(i) + random_double()) / (x_dimension-1);
                auto v = (double(j) + random_double()) / (y_dimension-1);
                Ray ray = camera.get_ray(u, v);
                glm::dvec3 x = color_of_ray(ray,world,max_depth,background);
                color += x;
            }
            
            auto r = color[0];
            auto g = color[1];
            auto b = color[2];

            // Divide the color by the number of samples and gamma-correct for gamma=2.0.
            auto scale = 1.0 / samples_per_pixel;
            r = sqrt(scale * r);
            g = sqrt(scale * g);
            b = sqrt(scale * b);
            
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
    
    auto aspect_ratio = 1.0;
    
    hittable_list world = cornell_box_pink();
//    auto background = glm::dvec3(0,0,0);
    auto lookfrom = glm::dvec3(0,3,5);
    auto lookat = glm::dvec3(0,0,0);
    auto vfov = 40.0;
    auto dist_to_focus = 10.0;
    auto aperture = 0.0;
    glm::dvec3 vup(0,1,0);
    
    
    Camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);
    
    // Render
    
    render(x_dimension, y_dimension,cam,world);
}
