__device__ float vector_length(float *v) {
    return sqrtf(v[0] * v[0] + v[1] * v[1]);
}
__device__ float vector_distance(float *a, float *b) {
    return sqrtf((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]));
}
__device__ void vector_normalize(float *v) {
    float length = vector_length(v);
    v[0] /= length;
    v[1] /= length;
}
__device__ void vector_add(float *a, float *b) {
    a[0] += b[0];
    a[1] += b[1];
}
__device__ float* vector_sub(float *a, float *b) {
    static float result[2];
    result[0] = a[0] - b[0];
    result[1] = a[1] - b[1];
    return result;
}

__global__ void vicsek_update(
        float *points, 
        float *directions,             
        float* new_points, 
        float* new_directions, 
        int n, 
        float radius, 
        float alpha, 
        float dt, 
        float *eta, 
        float width, 
        float height,
        float wall_repell_radius,
        float wall_repell_factor) {
        
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    float neighbors[32768]; 

    if (i < n) {                        
        int neighbors_count = 0;
        
        float dst;
        for (int j = 0; j < n; j++) {
            dst = vector_distance(&points[i * 2], &points[j * 2]);
            if (dst < radius) {                    
                neighbors[neighbors_count] = j;
                neighbors_count++;
            }
        }

        // check to avoid division by 0
        if (neighbors_count > 0) {   
                        
            // calculate average direction
            for (int j = 0; j < neighbors_count; j++) {
                int neighbour = neighbors[j] * 2;
                vector_add(&new_directions[i * 2], &directions[neighbour]);                    
            }                
            new_directions[i * 2]       /= neighbors_count;
            new_directions[i * 2 + 1]   /= neighbors_count;            

            // scale with alpha
            new_directions[i * 2]       = (1 - alpha) * directions[i * 2]       
                                        + alpha * new_directions[i * 2];
                                        
            new_directions[i * 2 + 1]   = (1 - alpha) * directions[i * 2 + 1]   
                                        + alpha * new_directions[i * 2 + 1];            
        }         
        else {
            // simply copy original direction into new array

            new_directions[i * 2]       = directions[i * 2];
            new_directions[i * 2 + 1]   = directions[i * 2 + 1];
        }      

        // wall repell
        new_directions[i * 2]       -= max((float) (points[i * 2] - width + wall_repell_radius), (float) 0) * wall_repell_factor * dt;
        new_directions[i * 2]       += max((float) (wall_repell_radius - points[i * 2]), (float) 0) * wall_repell_factor * dt;

        new_directions[i * 2 + 1]   -= max((float) (points[i * 2 + 1] - height + wall_repell_radius), (float) 0) * wall_repell_factor * dt;
        new_directions[i * 2 + 1]   += max((float) (wall_repell_radius - points[i * 2 + 1]), (float) 0) * wall_repell_factor * dt;

        // apply external forces
        vector_add(&new_directions[i * 2], &eta[i * 2]);            

        // normalize direction            
        vector_normalize(&new_directions[i * 2]);

        // update position
        new_points[i * 2]       = points[i * 2]     + new_directions[i * 2] * dt;
        new_points[i * 2 + 1]   = points[i * 2 + 1] + new_directions[i * 2 + 1] * dt;
    }
}