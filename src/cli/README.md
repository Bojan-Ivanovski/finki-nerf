# CLI Package

This package contains all command-line interface functionality for FINKI-NeRF.

## Structure

```
cli/
├── __init__.py           # Package exports
├── manager.py            # Main CLI manager and argument parsing
├── commands/             # Individual command handlers
│   ├── __init__.py       # Command handler exports
│   ├── base.py          # Abstract base class for all commands
│   ├── train.py         # Training command handler
│   ├── predict.py       # Single image prediction handler
│   ├── video.py         # Video generation handler
│   └── data.py          # Data inspection handler
└── README.md            # This file
```

## Key Components

### CLIManager (`manager.py`)
The main CLI coordinator that handles:
- Command-line argument parsing with argparse
- Command routing to appropriate handlers
- Global error handling and user feedback
- Help text generation and validation

### Command Handlers (`commands/`)
Individual command implementations that handle:
- **train**: Model training operations (list objects, generate data, train)
- **predict**: Single image generation from trained models
- **video**: Video sequence generation with multiple camera paths
- **data-length**: Dataset inspection and validation

Each command handler inherits from `CommandHandler` base class and implements the `execute(args)` method.

## Design Principles

1. **Separation of Concerns**: CLI logic is separate from business logic
2. **Modularity**: Each command is in its own file for maintainability
3. **Consistency**: All commands follow the same interface pattern
4. **Error Handling**: Comprehensive error handling with user-friendly messages
5. **Extensibility**: Easy to add new commands by creating new handler classes

## Adding New Commands

To add a new command:

1. Create a new handler file in `commands/` (e.g., `new_command.py`)
2. Inherit from `CommandHandler` and implement `execute(args)`
3. Add argument parser in `CLIManager._create_new_command_parser()`
4. Register handler in `CLIManager.__init__()`
5. Export in `commands/__init__.py`

## Usage

The CLI is used through the main entry point:

```bash
python main.py [command] [options]
```

Available commands:
- `train` - Train NeRF models
- `predict` - Generate single images
- `video` - Generate video sequences
- `data-length` - Inspect datasets

Each command has its own help available via:
```bash
python main.py [command] --help
```