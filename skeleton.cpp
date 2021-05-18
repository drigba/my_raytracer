#include "framework.h" 
 
const char* vertexSource = R"(
	#version 330
    precision highp float;
 
	uniform vec3 wLookAt, wRight, wUp;         
	layout(location = 0) in vec2 cCamWindowVertex;	
	out vec3 p;
 
	void main() {
		gl_Position = vec4(cCamWindowVertex, 0, 1);
		p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
	}
)";
 
const char* fragmentSource = R"(
	#version 330
    precision highp float;
 
	vec3 Le = vec3(2.1f,2.0f,2.0f);
	vec3 lightPosition = vec3(0.0f,0.0f,0.0f);
	const vec3 ka = vec3(0.5f,0.5f,0.5f);
	const float shininess = 300.0f;
	const int maxdepth = 5;
	const float epsilon = 0.01f;
	const float pi = 3.141593;
	struct Hit { 
		float t; 
		vec3 position, normal; 
		int mat;	
	};
 
	struct Ray {
		vec3 start, dir, weight;
	};
	struct Sphere {
		vec3 center;
		float radius;
	};
 
	const int objFaces = 12;
	uniform vec3 wEye, v[20];
	uniform int planes[objFaces*3];
	uniform vec3 kd[2], ks[2], F0;
	uniform Sphere sphere;
	Sphere sphere2;
	
	void getObjPlane(int i, float scale, out vec3 p, out vec3 normal){
		vec3 p1 = v[planes[3*i]-1],p2 = v[planes[3*i+1]-1], p3 = v[planes[3*i+2]-1];
		normal = cross(p2-p1, p3-p1);
		if(dot(p1,normal)<0) normal = -normal;
		p = p1 * scale + vec3(0,0,0.03f);
	}
 
	Hit intersectConvexPolyhedron(Ray ray, Hit hit, float scale, int mat){
		for(int i = 0; i < objFaces; i++){
			
			vec3 p1, normal;
			getObjPlane(i,scale,p1,normal); 
			float ti = abs(dot(normal,ray.dir))>epsilon ? dot(p1-ray.start, normal)/dot(normal,ray.dir):-1;
			if(ti<=epsilon || (ti > hit.t && hit.t > 0)) continue;
			vec3 pintersect = ray.start + ray.dir * ti;
			bool outside_small = false;
			bool outside_big = false;
			bool outside = false;
			for(int j = 0; j< objFaces; j++){ 
				if(i==j) continue;
				vec3 p11, n;
				getObjPlane(j,(scale-0.1f)/scale,p11,n);
				float myd = dot(n,pintersect-p11);
				if(myd>0){ 
					outside_small = true;
					break;
				}
				
			}
			if(outside_small){
				for(int j = 0; j< objFaces; j++){ 
					if(i==j) continue;
					vec3 p11, n;
					getObjPlane(j,scale,p11,n);
					float myd = dot(n,pintersect-p11);
					if(myd>0){ 
						outside = true;
						break;
					}	
				}
			}
			
			if(!outside){	
				if(!outside_small)
					hit.mat = 3;
				else
					hit.mat = 1;
				hit.t = ti;
				hit.position = pintersect;
				hit.normal = normalize(normal);
			}
		}
		return hit;
	}
	
 
 
	Hit intersect(const Sphere object, const Ray ray) {
		
		Hit hit;
        hit.t = -1;
        vec3 dist = ray.start - object.center;
        float para = 0.8;
        float parb = 0.6;
        float parc = 0.7;
 
        float a = para*(ray.dir.x * ray.dir.x) + parb*(ray.dir.y * ray.dir.y);        
        float b = 2*para*ray.start.x*ray.dir.x + 2*parb * ray.start.y*ray.dir.y-parc*ray.dir.z;
        float c = para * ray.start.x * ray.start.x + parb * ray.start.y * ray.start.y - parc* ray.start.z;
        
        float discr = b * b - 4.0f * a * c;
        if (discr < 0) return hit;
        float sqrt_discr = sqrt(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;
        float t2 = (-b - sqrt_discr) / 2.0f / a;
		if(t1<=0) return hit;
		Hit hit2;
		hit2.t = -1;
		vec3 pos1 = ray.start + t1*ray.dir;
		vec3 pos2 = ray.start + t2*ray.dir;
		float d1 = dot((pos1-object.center),(pos1-object.center));
		float d2 = dot((pos2-object.center),(pos2-object.center));
		if(d1 > object.radius && d2 >object.radius)
			return hit2;
		if(d1 > object.radius && d2<=object.radius){
			hit.t = t2;
			hit.position= pos2;
		}
		if(d1 <= object.radius && d2>object.radius){
			hit.t = t1;
			hit.position= pos1;
		}
		if(d1 <= object.radius && d2 < object.radius){
			if(d1< d2){
				hit.t = t1;
				hit.position= pos1;
			}
			else{
				hit.t = t2;
				hit.position = pos2;
			}
 
		}
 
 
 
		if(dot((hit.position-object.center),(hit.position-object.center)) > object.radius) return hit2;
		hit.normal = normalize(vec3((-2*para*hit.position.x/parc),(-2*parb*hit.position.y/parc),1));
		if(dot(hit.normal,ray.dir)>0)
			hit.normal = -1* hit.normal;
		hit.mat = 4;
        return hit;
	}
	
	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		bestHit.t = -1; 
		bestHit = intersect(sphere2, ray);
		bestHit = intersectConvexPolyhedron(ray,bestHit,1.0f,1); 
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}
vec3 Fresnel(vec3 F0, float cosTheta) { 
		return F0 + (vec3(1, 1, 1) - F0) * pow(cosTheta, 5);
	}
vec4 mult_q(vec4 q1, vec4 q2){
		vec4 q;
		q.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
		q.y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
		q.z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);
		q.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);
		return q;
}
	vec3 trace(Ray ray) {
		vec3 outRadiance = vec3(0.0,0.0,0.0); 
		vec3 outRadiance2 = vec3(0.5,0.5,0.5);
		for(int d = 0; d < maxdepth; d++) { 
			Hit hit = firstIntersect(ray); 
			if (hit.t < 0) break; 
			if (hit.mat < 2) { 
				vec3 lightdir = normalize(lightPosition - hit.position); 
				float cosTheta = dot(hit.normal, lightdir);			
				if (cosTheta > 0) {		
					
					vec3 LeIn = Le / dot(lightPosition - hit.position, lightPosition - hit.position); 
					outRadiance += ray.weight * LeIn * kd[hit.mat] * cosTheta; 
					
					vec3 halfway = normalize(-ray.dir + lightdir);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance += ray.weight * LeIn * ks[hit.mat] * pow(cosDelta, shininess); 
				}
				ray.weight *= ka; 
				break;
			}
		
			if(hit.mat == 4){
				ray.weight *= Fresnel(F0,1-dot(-ray.dir,hit.normal));
				outRadiance2 *= Fresnel(F0,1-dot(-ray.dir,hit.normal));
				d--;
			}
 
			ray.start = hit.position + hit.normal * epsilon; 
			ray.dir = reflect(ray.dir, hit.normal); 
			vec4 q1 = vec4(ray.dir,0);
			vec4 q2 = vec4(hit.normal.x*sin(36/180*pi), hit.normal.y* sin(36/180*pi),hit.normal.z*sin(36/180*pi),cos(36/180*pi));
			vec4 q3 = vec4(-1*hit.normal.x*sin(36/180*pi), -1*hit.normal.y* sin(36/180*pi),-1*hit.normal.z*sin(36/180*pi),cos(36/180*pi));
			vec4 q4;
			q1 = mult_q(q2,q1);
			q4 = mult_q(q1,q3);
			q1 = vec4(ray.start,0);
			q1 = mult_q(q2,q1);
			q1 = mult_q(q1,q3);
			if(hit.mat == 3){
				ray.dir = normalize(vec3(q4.x,q4.y,q4.z));
				ray.start = vec3(q1.x,q1.y,q1.z);			
			}
			
			
		}
 
	
		if(outRadiance == vec3(0,0,0))
			outRadiance = outRadiance2;
		return outRadiance;
	}
	
	in vec3 p; 
	out vec4 fragmentColor; 
 
	void main() {
		sphere2.center = vec3(0,0,0);
		sphere2.radius = 0.1 ;
		Ray ray; 
		ray.start = wEye; 
		ray.dir = normalize(p - wEye); 
		ray.weight = vec3(1,1,1); 
		fragmentColor = vec4(trace(ray), 1); 
	}
)";
 
 
 
 
 
struct Camera {
	
	vec3 eye, lookat, right, pvup, rvup; 
	float fov = 45 * (float)M_PI / 180;
public:
	Camera() : eye(0, 1, 1), pvup(0, 0, 1), lookat(0, 0, 0) { set(); }
	void set() {
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(pvup, w)) * f * tanf(fov / 2);
		rvup = normalize(cross(w, right)) * f * tanf(fov / 2);
	}
	void Animate(float t) {
		float r = sqrtf(eye.x * eye.x + eye.y * eye.y);
		eye = vec3(r * cos(t) + lookat.x, r * sin(t) + lookat.y, eye.z);
		set();
	}
};
 
 
 
 
 
 
 
Camera camera;
 
 
GPUProgram shader; 
 
 
float F(float n, float k) { return ((n - 1) * (n - 1) + k * k / ((n + 1) * (n + 1) + k * k)); } 
 
 
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	unsigned int vao, vbo; 
	glGenVertexArrays(1, &vao); glBindVertexArray(vao);
	glGenBuffers(1, &vbo);      glBindBuffer(GL_ARRAY_BUFFER, vbo);
	float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	 
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	shader.create(vertexSource, fragmentSource, "fragmentColor");
	float g = 0.618;
	float G = 1.618;
	std::vector<vec3> v = {
		vec3(0, g, G),
		vec3(0, -g, G),
		vec3(0, -g, -G),
		vec3(0, g, -G),
		vec3(G, 0, g),
		vec3(-G, 0, g),
		vec3(-G, 0, -g),
		vec3(G, 0, -g),
		vec3(g, G, 0),
		vec3(-g, G, 0),
		vec3(-g, -G, 0),
		vec3(g, -G, 0),
		vec3(1, 1, 1),
		vec3(-1, 1, 1),
		vec3(-1, -1, 1),
		vec3(1, -1, 1),
		vec3(1, -1, -1),
		vec3(1, 1, -1),
		vec3(-1, 1, -1),
		vec3(-1, -1, -1)
	}; 
	for (int i = 0; i < v.size(); i++) shader.setUniform(v[i], "v[" + std::to_string(i) + "]");
 
	std::vector<int> planes = {
		1,2,16, 1,13,9, 1,14,6, 2,15,11, 3,4,18, 3,17,12, 3,20,7, 19,10,9, 16,12,17, 5,8,18, 14,10,19, 6,7,20
	}; 
	for (int i = 0; i < planes.size(); i++) shader.setUniform(planes[i], "planes[" + std::to_string(i) + "]");
 
 
	shader.setUniform(vec3(1.0f, 0.5f, 0.31f), "kd[1]");
	shader.setUniform(vec3(1, 1, 1), "ks[1]");
 
	shader.setUniform(vec3(F(0.5, 3.1), F(0.35, 2.7), F(1.5, 1.9)), "F0"); 
	
 
	shader.Use();
}
 
 
void onDisplay() {
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
 
	shader.setUniform(camera.eye, "wEye");
	shader.setUniform(camera.lookat, "wLookAt");
	shader.setUniform(camera.right, "wRight");
	shader.setUniform(camera.pvup, "wUp");
 
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4); 
	glutSwapBuffers();
}
 
 
void onKeyboard(unsigned char key, int pX, int pY) {
	
}
 
 
void onKeyboardUp(unsigned char key, int pX, int pY) {
 
}
 
 
void onMouse(int button, int state, int pX, int pY) {
 
}
 
 
void onMouseMotion(int pX, int pY) {
}
 
 
void onIdle() {
 
		camera.Animate(glutGet(GLUT_ELAPSED_TIME) / 3000.0f);
	glutPostRedisplay();
}