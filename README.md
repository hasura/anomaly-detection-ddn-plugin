# Hasura DDN Anomaly Detection Plugin

A powerful hybrid anomaly detection system for Hasura DDN (Data Delivery Network) that combines statistical analysis with AI-powered query pattern detection to identify unusual data patterns, potential security concerns, and query anomalies in real-time.

## Features

- **Hybrid Detection System**
  - Statistical anomaly detection using Isolation Forest
  - AI-powered query pattern analysis using Claude
  - Real-time analysis of query results and patterns

- **Comprehensive Analysis**
  - Query pattern recognition
  - Statistical outlier detection
  - Historical data comparison
  - Security concern identification
  - Business logic validation

- **Persistent Storage**
  - Database-agnostic storage using SQLAlchemy
  - Model persistence and versioning
  - Configurable retention policies
  - Efficient data cleanup

- **Scalable Architecture**
  - Modular design
  - Configurable batch processing
  - Efficient resource utilization
  - Concurrent request handling

## Prerequisites

- Python 3.8+
- A relational database supported by SQLAlchemy:
  - PostgreSQL (reference implementation)
  - MySQL/MariaDB
  - SQLite
  - Oracle
  - Microsoft SQL Server
  - Or any database with SQLAlchemy dialect support
- Anthropic API key (for Claude integration)
- Hasura DDN instance

## Installation

This method will let you connect the anomaly detector plugin 
to your data sources. See "Adding to an Existing Supergraph"
for details on integrating with an existing supergraph.

1. Clone the repository:
```bash
git clone https://github.com/hasura/anomaly-detection-ddn-plugin
cd anomaly-detection-ddn-plugin
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install core dependencies:
```bash
pip install -r requirements.txt
```

4. Install database-specific driver:
```bash
# PostgreSQL
pip install psycopg2-binary

# MySQL
pip install mysqlclient

# Oracle
pip install cx_Oracle

# Microsoft SQL Server
pip install pyodbc
```

5. Set up environment variables:
```bash
cp .env.sample .env
cp example/.env.sample example/.env
```

* Remember to update the environment variables to working values.

6. Connect to your data
* Remove the chinook connector and connect to your data source.

```bash
ddn connector remove chinook
ddn connector init -i
```

## Adding to an Existing Supergraph
1. Follow steps 1-5 above.
2. In your existing supergraph compose.yaml, add:

```yaml
include:
  - path: ../compose.yaml
```

And, add the env variables into `globals/subgraph.yaml`, like this:

```yaml
kind: Subgraph
version: v2
definition:
  name: globals
  generator:
    rootPath: .
  includePaths:
    - metadata
  envMapping:
    ANOMALIES_URL:
      fromEnv: ANOMALIES_URL
    M_AUTH_KEY:
      fromEnv: M_AUTH_KEY
```
Set the path to the compose.yaml file in the directory where you 
installed the anomaly detector.

This will add the anomaly detector service container to the supergraph start up.

3. In your supergraph, copy the `example/globals/metadata/anomalies.hml` file to the same location within your supergraph.

This adds the plugin definition to the supergraph

4. In your supergraph .env file add.

```env
ANOMALIES_URL="http://local.hasura.dev:8787/anomalies"
M_AUTH_KEY=secret
ANTHROPIC_API_KEY=<do-your-own>
```

5. Start the supergraph

```bash
ddn run docker-start
```

This will create the anomaly tables.

6. Expose the anomaly tables into the same supergraph or to another of your choice so that you can run PromptQL against the results.

```bash
# Create the subgraph
ddn subgraph init data_quality
# Add it to the supergraph
ddn subgraph add --subgraph data_quality/subgraph.yaml --target-supergraph supergraph.yaml
# Create the data connector
ddn connector init --subgraph data_quality/subgraph.yaml -i
# Get its metadata
ddn connector introspect anomalies --subgraph data_quality/subgraph.yaml
# Add the tables to the subgraph
ddn model add anomalies "*" --subgraph data_quality/subgraph.yaml
ddn relationship add anomalies "*" --subgraph data_quality/subgraph.yaml
# Build and run the new supergraph
ddn supergraph build local
ddn run docker-start
```
## Environment Variables

Example `.env` configuration:

```bash
# Server Configuration
PORT=8787                  # Server port number
HOST=0.0.0.0              # Server host address

# Data Directory
ANOMALY_DETECTION_DATA=./tmp   # Directory for storing temporary files and data

# Anthropic Configuration
ANTHROPIC_API_KEY=your-api-key-here     # Your Anthropic API key
CLAUDE_MODEL=claude-3-7-sonnet-20250219 # Specific Claude model to use

# Anomaly Detection Configuration
MAX_RECORDS_PER_BATCH=50                # Maximum records to process in a single batch
HISTORICAL_RETENTION_DAYS=14            # Days to keep historical data
ANOMALY_RETENTION_DAYS=90               # Days to keep anomaly records
MODEL_RETENTION_DAYS=360                # Days to keep trained models
ANOMALY_THRESHOLD=0.1                   # Threshold for anomaly detection
MINIMUM_TRAINING_RECORDS=100            # Minimum records required for model training
MAX_HISTORICAL_RECORDS=100000           # Maximum historical records to store

# Processing Configuration
MAX_TOKENS=100000                      # Maximum tokens for LLM requests
MAX_RECORDS_PER_BATCH=50               # Maximum records to process in one batch
EXCLUDED_DATASETS=anomalies_.*,dq_.*       # Datasets matching these patterns are not processed

# Database Configuration
DB_HOST=your-db-host                  # Database host address
DB_CONNECT_ARGS=json-dict-of-args     # Optional - can be used to specify a schema
DB_PORT=5432                          # Database port
DB_NAME=anomalies                     # Database name
DB_USER=your-username                 # Database username
DB_PASSWORD=your-password             # Database password

# Logging
LOG_LEVEL=DEBUG                       # Logging level (DEBUG, INFO, WARNING, ERROR)

```


### Notes on Database Selection

- **PostgreSQL**: Reference implementation, recommended for production use
  - Robust JSON support
  - Advanced indexing capabilities
  - Excellent performance with large datasets

- **MySQL/MariaDB**: Good alternative
  - Wide adoption
  - Good performance
  - Some limitations with JSON operations

- **SQLite**: Suitable for development/testing
  - No separate server required
  - Limited concurrent access
  - Not recommended for production

- **Oracle/MSSQL**: Enterprise options
  - Good for integration with existing enterprise systems
  - Additional licensing considerations
  - May require specific configuration for optimal performance

## Usage


### API Endpoints

- `POST /anomalies`: Main anomaly detection endpoint
- `GET /health`: Health check endpoint
- `GET /history/<query_id>`: Get historical data for a query
- `DELETE /history/<query_id>`: Clear historical data
- `GET /model/<query_id>`: Get model information
- `POST /analyze/<query_id>`: Analyze a single record
- `GET /stats/<query_id>`: Get statistical summaries

## Example Directory

Check out the `example` directory for a complete working example:

```bash
# After updating .env and ./example/.env
python server.py
cd example
ddn run docker-start
```

Example request:
```bash
curl -X POST http://localhost:8787/anomalies \
  -H "Content-Type: application/json" \
  -H "X-Hasura-User: test-user" \
  -H "X-Hasura-Role: user" \
  -d @example/sample_request.json
```

Example response:
```json
{
  "results": {
    "users": {
      "is_anomaly": true,
      "score": -0.876,
      "model_details": {
        "model_type": "isolation_forest",
        "features_used": 5
      },
      "security_concerns": [
        {
          "type": "unusual_access_pattern",
          "severity": "medium"
        }
      ]
    }
  },
  "metadata": {
    "timestamp": "2024-11-10T10:30:00Z",
    "query_hash": "abc123",
    "total_records": 100,
    "total_anomalies": 3
  }
}
```

## Architecture

The system consists of several key components:

1. **Hybrid Detector** (`hybrid_detector.py`)
   - Combines statistical and AI-powered analysis
   - Manages analysis workflow
   - Coordinates between components

2. **Statistical Detector** (`statistical_detector.py`)
   - Implements Isolation Forest algorithm
   - Handles model training and persistence
   - Processes numerical features

3. **Query Detector** (`query_detector.py`)
   - AI-powered query analysis
   - Pattern recognition
   - Security validation

4. **Database Storage** (`db_storage.py`)
   - Manages data persistence
   - Handles model storage
   - Implements cleanup policies

## Roadmap

### Future Enhancements

Improvements from a simple sliding window to a more sophisticated seasonality approach.

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```
   Error: Could not connect to database
   ```
   - Verify database connection string format
   - Check if database driver is installed
   - Verify database credentials in `.env`
   - Ensure database server is running
   - Check network connectivity
   - Verify database permissions

2. **Database Driver Issues**
   ```
   Error: No module named 'psycopg2' (or similar)
   ```
   - Install appropriate database driver:
     ```bash
     # PostgreSQL
     pip install psycopg2-binary
     
     # MySQL
     pip install mysqlclient
     
     # Oracle
     pip install cx_Oracle
     
     # MSSQL
     pip install pyodbc
     ```

3. **API Key Issues**
   ```
   Error: Invalid API key
   ```
   - Verify ANTHROPIC_API_KEY in `.env`
   - Check API key permissions

4. **Model Training Errors**
   ```
   Error: Insufficient data for training
   ```
   - Ensure sufficient historical data
   - Check data format consistency
   - Verify database indexes

## Contributing

1. Fork the repository from https://github.com/hasura/anomaly-detection-ddn-plugin
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request to the main repository

### Development Guidelines

- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Include example usage
- Add meaningful commit messages

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hasura team for the DDN platform
- Anthropic for the Claude API
- scikit-learn team for the Isolation Forest implementation

## Support

For support, please:

1. Check the documentation
2. Review existing issues on [GitHub](https://github.com/hasura/anomaly-detection-ddn-plugin/issues)
3. Open a new issue with:
   - Detailed description
   - Steps to reproduce
   - System information
   - Relevant logs

## Security

To report security vulnerabilities, please follow Hasura's security policy or email security@hasura.io.

---

Made with ❤️ by Hasura
