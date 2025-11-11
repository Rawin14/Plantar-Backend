export interface User {
  id: string;
  email?: string;
  user_metadata?: {
    full_name?: string;
    avatar_url?: string;
  };
}

export interface Profile {
  id: string;
  nickname: string;
  age?: number;
  height?: number;
  weight?: number;
  gender?: 'male' | 'female' | 'other';
  birthdate?: string;
  created_at: string;
  updated_at: string;
}

export interface FootScan {
  id: string;
  user_id: string;
  foot_side: 'left' | 'right';
  images_url: string[];
  model_3d_url?: string;
  status: 'processing' | 'completed' | 'failed';
  measurements?: Measurements;
  processed_at?: string;
  created_at: string;
}

export interface Measurements {
  length: number;
  width: number;
  instep_height: number;
  arch_height: number;
  heel_width: number;
  ball_girth: number;
}

export interface Shoe {
  id: string;
  brand: string;
  model: string;
  category: 'running' | 'casual' | 'formal' | 'sports';
  sizes: string[];
  foot_shape_data?: any;
  image_url?: string;
  price?: number;
  created_at: string;
}

export interface ShoeRecommendation {
  id: string;
  scan_id: string;
  shoe_id?: string;
  shoe_name: string;
  brand: string;
  match_score: number;
  size: string;
  image_url?: string;
  price?: number;
  created_at: string;
}