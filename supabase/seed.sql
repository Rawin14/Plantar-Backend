-- ===== Seed Data for Shoes (รองเท้าที่ช่วยรองช้ำ) =====

INSERT INTO shoes (brand, model, category, arch_support_level, cushioning_level, heel_cup_depth, sizes, image_url, price, features) VALUES

-- Orthopedic Shoes
('Orthofeet', 'Coral Stretch Knit', 'orthopedic', 'high', 'firm', 'deep', 
 ARRAY['US 6', 'US 7', 'US 8', 'US 9', 'US 10', 'US 11'], 
 'https://images.unsplash.com/photo-1560343090-f0409e92791a', 4200.00,
 ARRAY['orthotic insole', 'arch support', 'heel cushion', 'extra depth']),

('Vionic', 'Walker Classic', 'orthopedic', 'high', 'medium', 'deep', 
 ARRAY['US 6', 'US 7', 'US 8', 'US 9', 'US 10'], 
 'https://images.unsplash.com/photo-1549298916-b41d501d3772', 3800.00,
 ARRAY['podiatrist designed', 'biomechanical footbed', 'shock absorption']),

-- Running Shoes with PF Support
('Brooks', 'Adrenaline GTS 23', 'running', 'high', 'medium', 'medium', 
 ARRAY['US 7', 'US 8', 'US 9', 'US 10', 'US 11'], 
 'https://images.unsplash.com/photo-1542291026-7eec264c27ff', 4800.00,
 ARRAY['GuideRails support', 'DNA LOFT cushioning', 'segmented crash pad']),

('ASICS', 'Gel-Kayano 29', 'running', 'high', 'soft', 'deep', 
 ARRAY['US 7', 'US 8', 'US 9', 'US 10', 'US 11'], 
 'https://images.unsplash.com/photo-1606107557195-0e29a4b5b4aa', 5200.00,
 ARRAY['GEL technology', 'LITETRUSS support', 'OrthoLite insole']),

('New Balance', 'Fresh Foam 860v12', 'running', 'medium', 'soft', 'medium', 
 ARRAY['US 7', 'US 8', 'US 9', 'US 10', 'US 11'], 
 'https://images.unsplash.com/photo-1539185441755-769473a23570', 4500.00,
 ARRAY['Fresh Foam midsole', 'medial post', 'dual-density foam']),

-- Casual Shoes with Support
('Hoka One One', 'Bondi 8', 'casual', 'medium', 'soft', 'medium', 
 ARRAY['US 7', 'US 8', 'US 9', 'US 10', 'US 11'], 
 'https://images.unsplash.com/photo-1608231387042-66d1773070a5', 5500.00,
 ARRAY['max cushioning', 'Meta-Rocker', 'breathable mesh']),

('Skechers', 'Arch Fit', 'casual', 'high', 'medium', 'deep', 
 ARRAY['US 6', 'US 7', 'US 8', 'US 9', 'US 10'], 
 'https://images.unsplash.com/photo-1595950653106-6c9ebd614d3a', 2800.00,
 ARRAY['podiatrist certified', 'removable insole', 'arch support']),

-- Medical Grade
('Dr. Comfort', 'Performance', 'medical', 'high', 'firm', 'deep', 
 ARRAY['US 6', 'US 7', 'US 8', 'US 9', 'US 10', 'US 11'], 
 'https://images.unsplash.com/photo-1543163521-1bf539c55dd2', 5800.00,
 ARRAY['diabetic approved', 'extra depth', 'gel inserts', 'seamless interior']),

('Propet', 'Stability Walker', 'medical', 'high', 'firm', 'deep', 
 ARRAY['US 6', 'US 7', 'US 8', 'US 9', 'US 10'], 
 'https://images.unsplash.com/photo-1600269452121-4f2416e55c28', 3500.00,
 ARRAY['Medicare approved', 'removable footbed', 'rigid heel counter']);

-- ===== Seed Exercise Data =====

-- Note: Exercise recommendations จะถูกสร้างโดย ML Service
-- ตามระดับความรุนแรงของแต่ละคน