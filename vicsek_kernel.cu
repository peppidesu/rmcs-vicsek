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

// from https://github.com/covexp/cuda-noise/blob/master/include/cuda_noise.cuh
__device__ unsigned int hash(unsigned int seed)
{
    seed = (seed + 0x7ed55d16) + (seed << 12);
    seed = (seed ^ 0xc761c23c) ^ (seed >> 19);
    seed = (seed + 0x165667b1) + (seed << 5);
    seed = (seed + 0xd3a2646c) ^ (seed << 9);
    seed = (seed + 0xfd7046c5) + (seed << 3);
    seed = (seed ^ 0xb55a4f09) ^ (seed >> 16);

    return seed;
}
// from https://github.com/covexp/cuda-noise/blob/master/include/cuda_noise.cuh
__device__ float randomFloat(unsigned int seed)
{
    unsigned int noiseVal = hash(seed);

    return ((float)noiseVal / (float)0xffffffff);
}

// from https://github.com/covexp/cuda-noise/blob/master/include/cuda_noise.cuh
__device__ float2 vectorNoise(float x, float y)
{
    return make_float2(
        randomFloat(x * 8231.0f + y * 34612.0f + 19283.0f) * 2.0f - 1.0f,
        randomFloat(x * 1171.0f + y * 9234.0f + 1466.0f) * 2.0f - 1.0f
        );
}

__global__ void vicsek_update(
        float*  points, 
        float*  directions,   
        bool*   alive_mask,          
        float*  new_points, 
        float*  new_directions,
        float*  predator_pos, 
        int     n, 
        float   radius, 
        float   detectionRadius,
        float   alpha, 
        float   damping,
        float   dt, 
        float   speed,
        float   noise_factor, 
        float   width, 
        float   height,
        float   wall_repell_radius,
        float   wall_repell_factor) {
        
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i < n) {                        

          
        new_points[i * 2]       = points[i * 2];
        new_points[i * 2 + 1]   = points[i * 2 + 1];

        if (!alive_mask[i]){
            new_directions[i * 2]       = directions[i * 2];
            new_directions[i * 2 + 1]   = directions[i * 2 + 1]; 
            return;
        } 

        bool isFleeing = vector_distance(&points[i * 2], predator_pos) < detectionRadius;
        float fleeing_direction[2] = {0,0};
        
        if (isFleeing) {
            fleeing_direction[0]    = points[i * 2] - predator_pos[0];
            fleeing_direction[1]    = points[i * 2 + 1] - predator_pos[1];
            vector_normalize(fleeing_direction);
        }  
        else {
            fleeing_direction[0]    = directions[i * 2];
            fleeing_direction[1]    = directions[i * 2 + 1];
        }


        
        // calculate average direction        
        float dst;
        int neighbors_count = 0;
        
        for (int j = 0; j < n; j++) {
            if (!alive_mask[j]) continue;

            dst = vector_distance(&points[i * 2], &points[j * 2]);
            if (dst > radius) continue;

            new_directions[i * 2] += directions[j * 2];
            new_directions[i * 2 + 1] += directions[j * 2 + 1];
            neighbors_count++;                   
                    
        }                
        if (neighbors_count == 0) {
            new_directions[i * 2]     = isFleeing ? fleeing_direction[0] : directions[i * 2];
            new_directions[i * 2 + 1] = isFleeing ? fleeing_direction[1] : directions[i * 2 + 1];
        }
        else {
            new_directions[i * 2]       /= neighbors_count;
            new_directions[i * 2 + 1]   /= neighbors_count;     
        }

        // scale with alpha
        new_directions[i * 2]       = (1 - alpha) * fleeing_direction[0]
                                    + alpha * new_directions[i * 2];
                                    
        new_directions[i * 2 + 1]   = (1 - alpha) * fleeing_direction[1]   
                                    + alpha * new_directions[i * 2 + 1];   
        float net_wall_factor = (1.0f / (float) wall_repell_radius) * wall_repell_factor * dt;
        // wall repell
        new_directions[i * 2]       -= max((float) (points[i * 2] - width + wall_repell_radius), 0.0f) 
                                        * net_wall_factor;
                                        
        new_directions[i * 2]       += max((float) (wall_repell_radius - points[i * 2]), 0.0f) 
                                        * net_wall_factor;

        new_directions[i * 2 + 1]   -= max((float) (points[i * 2 + 1] - height + wall_repell_radius), 0.0f) 
                                        * net_wall_factor;

        new_directions[i * 2 + 1]   += max((float) (wall_repell_radius - points[i * 2 + 1]), 0.0f) 
                                        * net_wall_factor;

        // apply noise
        float2 eta = vectorNoise(points[i * 2], points[i * 2 + 1]);
        new_directions[i * 2] += eta.x * noise_factor;            
        new_directions[i * 2 + 1] += eta.y * noise_factor;
        
        // damping
        new_directions[i * 2]       = (1 - damping) * directions[i * 2]
        + damping * new_directions[i * 2];
        
        new_directions[i * 2 + 1]   = (1 - damping) * directions[i * 2 + 1]   
        + damping * new_directions[i * 2 + 1];
        
        // normalize direction            
        vector_normalize(&new_directions[i * 2]);

        // update position
        new_points[i * 2] += new_directions[i * 2] * speed * dt;
        new_points[i * 2 + 1] += new_directions[i * 2 + 1] * speed * dt;
    }
}