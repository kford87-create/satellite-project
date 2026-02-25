
-- Run this in your Supabase SQL Editor
-- Dashboard → SQL Editor → New Query

-- Track every image in the dataset
CREATE TABLE IF NOT EXISTS dataset_images (
    id              BIGSERIAL PRIMARY KEY,
    stem            TEXT UNIQUE NOT NULL,
    split           TEXT NOT NULL,           -- 'train', 'val', 'test'
    prefix          TEXT NOT NULL,           -- 'spacenet', 'bootstrapped'
    image_path      TEXT NOT NULL,           -- Path in Supabase storage
    label_path      TEXT,                    -- Path to YOLO label file
    is_labeled      BOOLEAN DEFAULT TRUE,
    bootstrap_iteration INT DEFAULT 0,
    uncertainty_score   FLOAT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Track each bootstrapping iteration
CREATE TABLE IF NOT EXISTS bootstrap_iterations (
    id              BIGSERIAL PRIMARY KEY,
    iteration       INT NOT NULL,
    n_images_labeled INT,
    cumulative_labels INT,
    map50           FLOAT,
    map50_95        FLOAT,
    precision_score FLOAT,
    recall_score    FLOAT,
    fn_rate         FLOAT,
    map_gain_per_label FLOAT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Store false negative quantification reports
CREATE TABLE IF NOT EXISTS fn_reports (
    id              BIGSERIAL PRIMARY KEY,
    report_json     JSONB NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_dataset_images_labeled ON dataset_images(is_labeled);
CREATE INDEX IF NOT EXISTS idx_dataset_images_iteration ON dataset_images(bootstrap_iteration);
