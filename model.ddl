-- Graph Hierarchy
CREATE TABLE supergraph (
    name VARCHAR(255) PRIMARY KEY,
    version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE subgraph (
    name VARCHAR(255) PRIMARY KEY,
    description TEXT
);

CREATE TABLE subgraph_supergraph_map (
    subgraph_name VARCHAR(255) REFERENCES subgraph(name),
    supergraph_name VARCHAR(255) REFERENCES supergraph(name),
    PRIMARY KEY (subgraph_name, supergraph_name)
);

-- Authorization
CREATE TABLE role (
    name VARCHAR(255) PRIMARY KEY,
    supergraph_name VARCHAR(255) NOT NULL REFERENCES supergraph(name)
);

-- Data Sources
CREATE TABLE data_connector (
    name VARCHAR(255) PRIMARY KEY,
    subgraph_name VARCHAR(255) NOT NULL REFERENCES subgraph(name),
    read_url VARCHAR(1024) NOT NULL,
    write_url VARCHAR(1024) NOT NULL
);

-- Type System
CREATE TABLE scalar_type (
    name VARCHAR(255),
    connector_name VARCHAR(255) REFERENCES data_connector(name),
    representation_type VARCHAR(50) NOT NULL,  -- int16, int32, string, etc
    graphql_type_name VARCHAR(255) NOT NULL,
    PRIMARY KEY (name, connector_name)
);

CREATE TABLE scalar_type_operation (
    scalar_type_name VARCHAR(255),
    connector_name VARCHAR(255),
    operation_type VARCHAR(50) NOT NULL,      -- aggregate_function, comparison_operator
    operation_name VARCHAR(255) NOT NULL,
    return_type VARCHAR(255) NOT NULL,
    PRIMARY KEY (scalar_type_name, connector_name, operation_type, operation_name),
    FOREIGN KEY (scalar_type_name, connector_name)
        REFERENCES scalar_type(name, connector_name)
);

CREATE TABLE object_type (
    name VARCHAR(255) PRIMARY KEY,
    connector_name VARCHAR(255) NOT NULL REFERENCES data_connector(name),
    description TEXT,
    graphql_type_name VARCHAR(255) NOT NULL,
    graphql_input_type_name VARCHAR(255)
);

CREATE TABLE object_field (
    object_type_name VARCHAR(255) REFERENCES object_type(name),
    logical_field_name VARCHAR(255),
    scalar_type_name VARCHAR(255),
    connector_name VARCHAR(255),
    description TEXT,
    is_nullable BOOLEAN NOT NULL DEFAULT false,
    PRIMARY KEY (object_type_name, logical_field_name),
    FOREIGN KEY (scalar_type_name, connector_name)
        REFERENCES scalar_type(name, connector_name)
);

-- Collections and Physical Schema
CREATE TABLE collection (
    name VARCHAR(255) PRIMARY KEY,
    connector_name VARCHAR(255) NOT NULL REFERENCES data_connector(name),
    description TEXT,
    object_type_name VARCHAR(255) NOT NULL REFERENCES object_type(name),
    physical_collection_name VARCHAR(255) NOT NULL
);

CREATE TABLE collection_field (
    collection_name VARCHAR(255) REFERENCES collection(name),
    physical_field_name VARCHAR(255),
    scalar_type_name VARCHAR(255),
    connector_name VARCHAR(255),
    is_nullable BOOLEAN NOT NULL DEFAULT false,
    PRIMARY KEY (collection_name, physical_field_name),
    FOREIGN KEY (scalar_type_name, connector_name)
        REFERENCES scalar_type(name, connector_name)
);

-- Physical to Logical Mapping
CREATE TABLE field_map (
    collection_name VARCHAR(255),
    physical_field_name VARCHAR(255),
    object_type_name VARCHAR(255),
    logical_field_name VARCHAR(255),
    PRIMARY KEY (collection_name, physical_field_name, object_type_name, logical_field_name),
    FOREIGN KEY (collection_name, physical_field_name)
        REFERENCES collection_field(collection_name, physical_field_name),
    FOREIGN KEY (object_type_name, logical_field_name)
        REFERENCES object_field(object_type_name, logical_field_name)
);

-- Relationships
CREATE TABLE relationship (
    name VARCHAR(255),
    subgraph_name VARCHAR(255) REFERENCES subgraph(name),
    source_type_name VARCHAR(255) REFERENCES object_type(name),
    target_type_name VARCHAR(255) REFERENCES object_type(name),
    relationship_type VARCHAR(50) NOT NULL,   -- Object or Array
    graphql_field_name VARCHAR(255),
    PRIMARY KEY (name, subgraph_name)
);

CREATE TABLE relationship_field_pair (
    relationship_name VARCHAR(255),
    subgraph_name VARCHAR(255),
    source_type_name VARCHAR(255),
    source_field_name VARCHAR(255),
    target_type_name VARCHAR(255),
    target_field_name VARCHAR(255),
    PRIMARY KEY (relationship_name, subgraph_name, source_field_name),
    FOREIGN KEY (relationship_name, subgraph_name)
        REFERENCES relationship(name, subgraph_name),
    FOREIGN KEY (source_type_name, source_field_name)
        REFERENCES object_field(object_type_name, logical_field_name),
    FOREIGN KEY (target_type_name, target_field_name)
        REFERENCES object_field(object_type_name, logical_field_name)
);

-- Permissions
CREATE TABLE type_permission (
    subgraph_name VARCHAR(255) REFERENCES subgraph(name),
    type_name VARCHAR(255) REFERENCES object_type(name),
    role_name VARCHAR(255) REFERENCES role(name),
    operation_type VARCHAR(50) NOT NULL,      -- input, output
    PRIMARY KEY (subgraph_name, type_name, role_name, operation_type)
);

CREATE TABLE allowed_field (
    subgraph_name VARCHAR(255),
    type_name VARCHAR(255),
    role_name VARCHAR(255),
    object_type_name VARCHAR(255),
    field_name VARCHAR(255),
    PRIMARY KEY (subgraph_name, type_name, role_name, field_name),
    FOREIGN KEY (subgraph_name, type_name, role_name)
        REFERENCES type_permission(subgraph_name, type_name, role_name),
    FOREIGN KEY (object_type_name, field_name)
        REFERENCES object_field(object_type_name, logical_field_name)
);

-- Query Capabilities
CREATE TABLE query_capability (
    name VARCHAR(255),
    subgraph_name VARCHAR(255) REFERENCES subgraph(name),
    capability_type VARCHAR(50) NOT NULL,     -- aggregate, nested_fields, etc
    is_enabled BOOLEAN NOT NULL DEFAULT false,
    PRIMARY KEY (name, subgraph_name)
);

-- GraphQL Configuration
CREATE TABLE graphql_config (
    key VARCHAR(255),
    subgraph_name VARCHAR(255) REFERENCES subgraph(name),
    value JSON NOT NULL,
    PRIMARY KEY (key, subgraph_name)
);

-- Function and Procedure Support
CREATE TABLE function (
    name VARCHAR(255) PRIMARY KEY,
    connector_name VARCHAR(255) NOT NULL REFERENCES data_connector(name),
    description TEXT,
    return_type_name VARCHAR(255) NOT NULL,
    return_type_connector VARCHAR(255) NOT NULL,
    FOREIGN KEY (return_type_name, return_type_connector)
        REFERENCES scalar_type(name, connector_name)
);

CREATE TABLE procedure (
    name VARCHAR(255) PRIMARY KEY,
    connector_name VARCHAR(255) NOT NULL REFERENCES data_connector(name),
    description TEXT,
    result_type_name VARCHAR(255) NOT NULL REFERENCES object_type(name)
);

-- Arguments
CREATE TABLE argument (
    parent_type VARCHAR(50) NOT NULL,        -- function, procedure
    parent_name VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    scalar_type_name VARCHAR(255) NOT NULL,
    connector_name VARCHAR(255) NOT NULL,
    description TEXT,
    is_required BOOLEAN NOT NULL DEFAULT false,
    PRIMARY KEY (parent_type, parent_name, name),
    FOREIGN KEY (scalar_type_name, connector_name)
        REFERENCES scalar_type(name, connector_name)
);
