#!/usr/bin/env python3
"""
FINKI-NeRF Main Entry Point

This is the main entry point for the FINKI-NeRF command-line interface.
It provides a minimal bootstrap that loads the CLI manager and handles
basic initialization and error handling.

For the full implementation details, see the cli/ and core/ packages.
"""

import sys
from dotenv import load_dotenv
from logs import logger
from cli import CLIManager

# Load environment variables
load_dotenv()


def main():
    """Main entry point for FINKI-NeRF CLI application.

    This function:
    1. Creates the CLI manager
    2. Parses command-line arguments
    3. Sets up logging
    4. Executes the requested command
    5. Handles errors and user interruptions
    """
    # Create CLI manager and parse arguments
    cli_manager = CLIManager()
    parser = cli_manager.create_parser()
    args = parser.parse_args()

    # Configure logging
    logger.setLevel(args.log_level)

    try:
        # Execute the requested command
        cli_manager.execute_command(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
