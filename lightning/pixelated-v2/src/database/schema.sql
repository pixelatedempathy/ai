-- Database schema for Pixelated Empathy

-- Users table
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(255) PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    preferences JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

-- Conversation sessions
CREATE TABLE IF NOT EXISTS conversation_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) REFERENCES users(user_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Individual messages within conversations
CREATE TABLE IF NOT EXISTS conversation_messages (
    message_id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) REFERENCES conversation_sessions(session_id),
    role VARCHAR(50) NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Model responses with style information
CREATE TABLE IF NOT EXISTS model_responses (
    response_id SERIAL PRIMARY KEY,
    message_id INTEGER REFERENCES conversation_messages(message_id),
    content TEXT NOT NULL,
    confidence FLOAT NOT NULL,
    primary_style VARCHAR(50) NOT NULL,
    style_scores JSONB NOT NULL,
    processing_time FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Training examples
CREATE TABLE IF NOT EXISTS training_examples (
    example_id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    style VARCHAR(50) NOT NULL,
    source VARCHAR(255) NOT NULL,
    quality_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_conversation_sessions_user_id ON conversation_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_session_id ON conversation_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_timestamp ON conversation_messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_model_responses_message_id ON model_responses(message_id);
CREATE INDEX IF NOT EXISTS idx_training_examples_style ON training_examples(style);
CREATE INDEX IF NOT EXISTS idx_training_examples_source ON training_examples(source);

-- Update trigger for conversation_sessions
CREATE OR REPLACE FUNCTION update_conversation_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_conversation_updated_at
    BEFORE UPDATE ON conversation_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_conversation_updated_at();
