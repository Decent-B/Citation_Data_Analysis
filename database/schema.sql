-- PostgreSQL Schema for Citation Data
-- Three normalized tables: papers, paper_references, related_works

-- Papers table: Core paper metadata
CREATE TABLE IF NOT EXISTS papers (
    id VARCHAR(20) PRIMARY KEY,
    doi TEXT,
    title TEXT,
    apc_list_price INTEGER,
    topic VARCHAR(20),
    referenced_works_count INTEGER DEFAULT 0,
    cited_by_count INTEGER DEFAULT 0,
    publication_date DATE,
    authors JSONB  -- Array of author IDs
);

-- Paper references table: Citation relationships
-- Which paper cites which paper
CREATE TABLE IF NOT EXISTS paper_references (
    citing_paper_id VARCHAR(20) NOT NULL,
    cited_paper_id VARCHAR(20) NOT NULL,
    PRIMARY KEY (citing_paper_id, cited_paper_id)
);

-- Related works table: Related paper relationships
CREATE TABLE IF NOT EXISTS related_works (
    paper_id VARCHAR(20) NOT NULL,
    related_paper_id VARCHAR(20) NOT NULL,
    PRIMARY KEY (paper_id, related_paper_id)
);

-- Indexes for query performance (created after bulk insert for faster migration)
-- Uncomment these after migration is complete, or run separately

-- CREATE INDEX idx_papers_topic ON papers(topic);
-- CREATE INDEX idx_papers_publication_date ON papers(publication_date);
-- CREATE INDEX idx_papers_cited_by_count ON papers(cited_by_count DESC);
-- CREATE INDEX idx_paper_references_cited ON paper_references(cited_paper_id);
-- CREATE INDEX idx_related_works_related ON related_works(related_paper_id);
