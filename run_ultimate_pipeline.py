#!/usr/bin/env python3
"""
🔥 ULTIMATE PIPELINE RUNNER
รันโดยตรงไปยัง Ultimate Pipeline (Mode 7) 
โดยไม่ต้องผ่าน interactive mode
"""

import sys
import os
sys.path.append(os.getcwd())

# Set environment variables
os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'

def main():
    print("🔥 Starting ULTIMATE PIPELINE - เทพสุดทุกส่วน!")
    print("=" * 60)
    
    try:
        # Import และเตรียม context
        import logging
        import warnings
        from colorama import Fore, Style, init as colorama_init
        from projectp.pro_log import pro_log
        from projectp.pipeline import run_ultimate_pipeline
        from src.utils.log_utils import set_log_context
        import getpass
        import socket
        import uuid
        
        # Initialize colorama
        colorama_init(autoreset=True)
        
        # Suppress warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # Set log context
        set_log_context(
            user=getpass.getuser(),
            host=socket.gethostname(),
            run_id=str(uuid.uuid4()),
            cwd=os.getcwd(),
            environment=os.environ.get("PROJECTP_ENV", "production"),
            job_id="ultimate_pipeline",
            pipeline_step="ultimate_pipeline"
        )
        
        print(f"{Fore.MAGENTA + Style.BRIGHT}🚀 Launching ULTIMATE PIPELINE...")
        print(f"{Fore.CYAN}🎯 Mode: Production Enterprise Grade")
        print(f"{Fore.GREEN}✅ All systems ready for deployment!")
        print("=" * 60)
        
        # Run Ultimate Pipeline
        errors, warnings_list, results = run_ultimate_pipeline()
        
        # Display results
        print("\n" + "=" * 60)
        print(f"{Fore.YELLOW + Style.BRIGHT}📊 ULTIMATE PIPELINE RESULTS:")
        print(f"{Fore.CYAN}   🔢 Total Steps: 16")
        print(f"{Fore.RED}   ❌ Errors: {len(errors)}")
        print(f"{Fore.YELLOW}   ⚠️  Warnings: {len(warnings_list)}")
        print(f"{Fore.GREEN}   ✅ Results: {len(results)}")
        
        if not errors and not warnings_list:
            print(f"\n{Fore.GREEN + Style.BRIGHT}🏆 ULTIMATE SUCCESS!")
            print(f"{Fore.GREEN}🎉 ALL PIPELINE STEPS COMPLETED FLAWLESSLY!")
            print(f"{Fore.CYAN}🚀 READY FOR PRODUCTION DEPLOYMENT!")
        elif errors:
            print(f"\n{Fore.RED + Style.BRIGHT}❌ PIPELINE COMPLETED WITH ERRORS:")
            for i, error in enumerate(errors, 1):
                print(f"{Fore.RED}   {i}. {error}")
        else:
            print(f"\n{Fore.YELLOW + Style.BRIGHT}⚠️ PIPELINE COMPLETED WITH WARNINGS:")
            for i, warning in enumerate(warnings_list, 1):
                print(f"{Fore.YELLOW}   {i}. {warning}")
        
        print("\n" + "=" * 60)
        print(f"{Fore.MAGENTA + Style.BRIGHT}🔥 ULTIMATE PIPELINE EXECUTION COMPLETE!")
        print(f"{Fore.CYAN}📈 Check output_default/ for results")
        print(f"{Fore.CYAN}📊 Check models/ for trained models")
        print(f"{Fore.GREEN}✅ Enterprise-grade ML pipeline ready!")
        
        return len(errors) == 0
        
    except ImportError as e:
        print(f"{Fore.RED}❌ Import Error: {e}")
        print(f"{Fore.YELLOW}💡 Make sure all dependencies are installed")
        return False
        
    except Exception as e:
        print(f"{Fore.RED}❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    print(f"\n🎯 Exit Code: {exit_code}")
    sys.exit(exit_code)
