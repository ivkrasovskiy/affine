# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup
```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup virtual environment and install dependencies
uv venv && source .venv/bin/activate && uv pip install -e .

# Verify installation
af
```

### Docker Development
```bash
# Build and push Docker image
./build_and_push.sh

# Run validator with Docker Compose (production)
docker-compose down && docker-compose pull && docker-compose up -d && docker-compose logs -f

# Run with local build
docker compose -f docker-compose.yml -f docker-compose.local.yml down --remove-orphans
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --build --remove-orphans
docker compose -f docker-compose.yml -f docker-compose.local.yml logs -f
```

### Core Commands
```bash
# Main CLI (Affine operations)
af -vv validate        # Run validator locally
af -vvv pull <uid> --model_path <path>   # Pull model from network
af -vvv push --coldkey <cold> --hotkey <hot> --model_path <path>  # Push model

# Quixand CLI (Sandbox operations)
qs sandbox create --template <template>    # Create sandbox
qs sandbox exec <id> <command>            # Execute command in sandbox
qs sandbox kill <id>                      # Kill sandbox
qs files put <id> <local> <remote>        # Upload file to sandbox
qs files get <id> <remote> <local>        # Download file from sandbox
```

## Architecture

### Core Components

**Main Affine System**: The root system provides mining/validation infrastructure for Bittensor network (subnet 120). Key modules:
- `affine/__init__.py` - Main entry point with CLI, SDK functions, and environment factories
- `affine/tasks.py` - Core task evaluation environments (SAT, ABD, DED, HVM, ELR, AgentGym variants)
- `affine/envs/` - Environment-specific implementations (abd.py, ded.py, elr.py, hvm.py, sat.py)

**Quixand System**: Embedded sandbox management system for code execution:
- `affine/quixand/core/` - Core sandbox functionality (sandbox.py, lifecycle.py, processes.py)
- `affine/quixand/adapters/` - Runtime adapters (Docker, Podman, HTTP endpoints)
- `affine/quixand/container/` - Container runtime implementations
- `affine/quixand/cli/main.py` - Quixand CLI interface

### Key Patterns

**Environment System**: Dual-layer environment architecture:
- `BaseSDKEnv` - Abstract base for all environments
- `AffineSDKEnv` - Affine-specific environments (SAT, ABD, DED, etc.)
- `AgentGymSDKEnv` - AgentGym environments (ALFWORLD, WEBSHOP, etc.)
- Environment registry pattern for dynamic instantiation

**Sandbox Management**: 
- Template-based sandbox creation (`affine:<env>`, `agentgym:<env>`)
- Shared sandbox instances for efficiency
- Proxy-based evaluation with configurable timeouts

**Async SDK Pattern**: All evaluation methods are async and support both single miners and batch operations.

### Configuration

**Environment Variables** (see `.env.example`):
- `CHUTES_API_KEY` - Required for Chutes platform access
- `BT_WALLET_COLD/HOT` - Bittensor wallet names
- `HF_USER/TOKEN` - Hugging Face credentials for miners
- `SUBTENSOR_ENDPOINT` - Bittensor network endpoint
- `AFFINE_ENV_LIST` - Comma-separated list of environments to evaluate

**Docker Services**:
- `validator` - Main validation service
- `runner` - Separate runner service  
- `signer` - Wallet signing service
- `grafana` - Monitoring dashboard
- `watchtower` - Auto-update service

### Development Notes

- Uses `uv` for Python package management instead of pip
- Prometheus metrics integration on port 8000
- Redis caching for blockchain data
- Supports both local development and containerized deployment
- Environment templates stored in `affine/quixand/env_templates/`