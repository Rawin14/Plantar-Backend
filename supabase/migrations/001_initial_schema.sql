-- ===== Enable Extensions =====
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ===== Users Profile Table =====
CREATE TABLE profiles (
  id UUID REFERENCES auth.users ON DELETE CASCADE PRIMARY KEY,
  nickname TEXT NOT NULL,
  age INTEGER,
  height DECIMAL(5,2),
  weight DECIMAL(5,2),
  gender TEXT CHECK (gender IN ('male', 'female', 'other')),
  birthdate DATE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW())
);

-- ===== Foot Scans Table (ปรับใหม่) =====
CREATE TABLE foot_scans (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES auth.users ON DELETE CASCADE NOT NULL,
  foot_side TEXT NOT NULL CHECK (foot_side IN ('left', 'right')),
  
  -- Images
  images_url TEXT[] NOT NULL,
  
  -- 3D Model (Optional)
  model_3d_url TEXT,
  
  -- Plantar Fasciitis Assessment
  pf_severity TEXT CHECK (pf_severity IN ('low', 'medium', 'high')),
  pf_score DECIMAL(5,2) CHECK (pf_score >= 0 AND pf_score <= 100),
  
  -- Foot Analysis
  foot_analysis JSONB, -- ข้อมูลการวิเคราะห์เท้า
  arch_type TEXT CHECK (arch_type IN ('flat', 'normal', 'high')),
  pressure_points JSONB, -- จุดกดทับ
  
  -- Processing
  status TEXT DEFAULT 'processing' CHECK (status IN ('processing', 'completed', 'failed')),
  processed_at TIMESTAMP WITH TIME ZONE,
  error_message TEXT,
  
  created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW())
);

-- ===== Plantar Fasciitis Indicators =====
CREATE TABLE pf_indicators (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  scan_id UUID REFERENCES foot_scans ON DELETE CASCADE NOT NULL,
  
  -- Indicators (0-100 score each)
  arch_collapse_score DECIMAL(5,2),
  heel_pain_index DECIMAL(5,2),
  pressure_distribution_score DECIMAL(5,2),
  foot_alignment_score DECIMAL(5,2),
  flexibility_score DECIMAL(5,2),

  scan_part_score DECIMAL(5,2),
  questionnaire_part_score DECIMAL(5,2),
  bmi_score DECIMAL(5,2),
  
  -- Details
  risk_factors TEXT[],
  recommendations TEXT[],
  
  created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW())
);

-- ===== Exercise Recommendations =====
CREATE TABLE exercise_recommendations (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  scan_id UUID REFERENCES foot_scans ON DELETE CASCADE NOT NULL,
  
  exercise_name TEXT NOT NULL,
  description TEXT,
  video_url TEXT,
  duration_minutes INTEGER,
  difficulty TEXT CHECK (difficulty IN ('easy', 'medium', 'hard')),
  recommended_frequency TEXT, -- "2 times per day"
  
  created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW())
);

-- ===== Shoe Recommendations (ปรับใหม่) =====
CREATE TABLE shoe_recommendations (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  scan_id UUID REFERENCES foot_scans ON DELETE CASCADE NOT NULL,
  
  shoe_name TEXT NOT NULL,
  brand TEXT NOT NULL,
  category TEXT CHECK (category IN ('orthopedic', 'running', 'casual', 'medical')),
  
  -- Matching
  match_score DECIMAL(5,2) CHECK (match_score >= 0 AND match_score <= 100),
  pf_support_score DECIMAL(5,2), -- คะแนนช่วยรองช้ำ
  
  -- Details
  size_recommendation TEXT,
  arch_support_level TEXT CHECK (arch_support_level IN ('low', 'medium', 'high')),
  cushioning_level TEXT CHECK (cushioning_level IN ('soft', 'medium', 'firm')),
  
  image_url TEXT,
  price DECIMAL(10,2),
  purchase_link TEXT,
  
  created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW())
);

-- ===== Shoes Database (ปรับใหม่) =====
CREATE TABLE shoes (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  brand TEXT NOT NULL,
  model TEXT NOT NULL,
  category TEXT CHECK (category IN ('orthopedic', 'running', 'casual', 'medical')),
  
  -- Plantar Fasciitis Features
  arch_support_level TEXT CHECK (arch_support_level IN ('low', 'medium', 'high')),
  cushioning_level TEXT CHECK (cushioning_level IN ('soft', 'medium', 'firm')),
  heel_cup_depth TEXT CHECK (heel_cup_depth IN ('shallow', 'medium', 'deep')),
  
  -- Sizing
  sizes TEXT[] NOT NULL,
  
  -- Additional Info
  foot_shape_data JSONB,
  image_url TEXT,
  price DECIMAL(10,2),
  features TEXT[], -- ["arch support", "shock absorption", "heel cushion"]
  
  created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW())
);

-- ===== Indexes =====
CREATE INDEX idx_foot_scans_user_id ON foot_scans(user_id);
CREATE INDEX idx_foot_scans_status ON foot_scans(status);
CREATE INDEX idx_foot_scans_pf_severity ON foot_scans(pf_severity);
CREATE INDEX idx_pf_indicators_scan_id ON pf_indicators(scan_id);
CREATE INDEX idx_exercise_recommendations_scan_id ON exercise_recommendations(scan_id);
CREATE INDEX idx_shoe_recommendations_scan_id ON shoe_recommendations(scan_id);
CREATE INDEX idx_shoe_recommendations_match_score ON shoe_recommendations(match_score DESC);
CREATE INDEX idx_shoes_category ON shoes(category);
CREATE INDEX idx_shoes_arch_support ON shoes(arch_support_level);

-- ===== Row Level Security =====
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE foot_scans ENABLE ROW LEVEL SECURITY;
ALTER TABLE pf_indicators ENABLE ROW LEVEL SECURITY;
ALTER TABLE exercise_recommendations ENABLE ROW LEVEL SECURITY;
ALTER TABLE shoe_recommendations ENABLE ROW LEVEL SECURITY;
ALTER TABLE shoes ENABLE ROW LEVEL SECURITY;

-- Policies for profiles
CREATE POLICY "Users can view own profile"
  ON profiles FOR SELECT
  USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
  ON profiles FOR UPDATE
  USING (auth.uid() = id);

CREATE POLICY "Users can insert own profile"
  ON profiles FOR INSERT
  WITH CHECK (auth.uid() = id);

-- Policies for foot_scans
CREATE POLICY "Users can view own scans"
  ON foot_scans FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own scans"
  ON foot_scans FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own scans"
  ON foot_scans FOR DELETE
  USING (auth.uid() = user_id);

-- Policies for pf_indicators
CREATE POLICY "Users can view own indicators"
  ON pf_indicators FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM foot_scans
      WHERE foot_scans.id = pf_indicators.scan_id
      AND foot_scans.user_id = auth.uid()
    )
  );

-- Policies for exercise_recommendations
CREATE POLICY "Users can view own exercises"
  ON exercise_recommendations FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM foot_scans
      WHERE foot_scans.id = exercise_recommendations.scan_id
      AND foot_scans.user_id = auth.uid()
    )
  );

-- Policies for shoe_recommendations
CREATE POLICY "Users can view own shoe recommendations"
  ON shoe_recommendations FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM foot_scans
      WHERE foot_scans.id = shoe_recommendations.scan_id
      AND foot_scans.user_id = auth.uid()
    )
  );

-- Policies for shoes (public read)
CREATE POLICY "Anyone can view shoes"
  ON shoes FOR SELECT
  TO authenticated, anon
  USING (true);

-- ===== Functions =====

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = TIMEZONE('utc', NOW());
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_profiles_updated_at
    BEFORE UPDATE ON profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ===== Comments =====
COMMENT ON TABLE profiles IS 'User profile information';
COMMENT ON TABLE foot_scans IS 'Foot scan data and plantar fasciitis assessment';
COMMENT ON TABLE pf_indicators IS 'Detailed plantar fasciitis indicators';
COMMENT ON TABLE exercise_recommendations IS 'Exercise recommendations for treatment';
COMMENT ON TABLE shoe_recommendations IS 'Shoe recommendations for plantar fasciitis';
COMMENT ON TABLE shoes IS 'Shoe database with PF support features';