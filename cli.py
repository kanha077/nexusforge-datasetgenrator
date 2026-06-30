import argparse
from pipeline.orchestrator import Orchestrator

def main():
    parser = argparse.ArgumentParser(description="NexusForge Local Dataset Generation")
    parser.add_argument("--config", type=str, default="configs", help="Directory containing configuration yaml files")
    args = parser.parse_args()
    
    print(f"Starting NexusForge with configs from {args.config}")
    orchestrator = Orchestrator(config_dir=args.config)
    orchestrator.run()

if __name__ == "__main__":
    main()
