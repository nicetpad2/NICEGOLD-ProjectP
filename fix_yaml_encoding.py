#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YAML Encoding Fixer
Fix encoding issues in all YAML configuration files
"""

import os
import glob
import codecs
import yaml
from typing import List, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yaml_encoding_fix.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def try_encodings(file_path: str) -> Optional[str]:
    """Try different encodings to find the correct one"""
    encodings = ['utf-8', 'cp1252', 'iso-8859-1', 'latin-1', 'ascii']
    
    for encoding in encodings:
        try:
            with codecs.open(file_path, 'r', encoding=encoding) as f:
                f.read()
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    return None

def convert_to_utf8(file_path: str) -> bool:
    """Convert a file to UTF-8 encoding"""
    try:
        # Try to find working encoding
        current_encoding = try_encodings(file_path)
        if not current_encoding:
            logger.error(f"Could not read {file_path} with any encoding")
            return False
            
        logger.info(f"Processing {file_path} (current encoding: {current_encoding})")
        
        # Read file with detected encoding
        with codecs.open(file_path, 'r', encoding=current_encoding) as f:
            content = f.read()
        
        # Validate YAML content (if possible)
        try:
            yaml.safe_load(content)
        except yaml.YAMLError as e:
            logger.warning(f"YAML syntax issue in {file_path}: {e}")
            # Continue anyway - might just be a template
        
        # Write as UTF-8
        with codecs.open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Successfully converted {file_path} to UTF-8")
        return True
        
    except Exception as e:
        logger.error(f"Error converting {file_path}: {e}")
        return False

def find_yaml_files() -> List[str]:
    """Find all YAML files in the project (excluding venv directories)"""
    yaml_files = []
    
    # Search patterns
    patterns = [
        '*.yaml',
        '*.yml', 
        'config/*.yaml',
        'config/*.yml',
        'configs/*.yaml',
        'configs/*.yml',
        'protection_config/*.yaml',
        'protection_config/*.yml',
        'agent/*.yaml',
        'agent/*.yml',
        'templates/*.yaml',
        'templates/*.yml',
        'tests/fixtures/*.yaml',
        'tests/fixtures/*.yml',
        'k8s/*.yaml',
        'k8s/*.yml'
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        yaml_files.extend(files)
    
    # Filter out virtual environment directories
    filtered_files = []
    for file_path in yaml_files:
        if not any(venv_dir in file_path for venv_dir in ['venv', 'env', '.env', 'site-packages']):
            filtered_files.append(file_path)
    
    # Remove duplicates and sort
    filtered_files = sorted(list(set(filtered_files)))
    
    return filtered_files

def main():
    """Main function to fix YAML encoding issues"""
    logger.info("Starting YAML encoding fix...")
    
    # Find all YAML files
    yaml_files = find_yaml_files()
    logger.info(f"Found {len(yaml_files)} YAML files to process")
    
    success_count = 0
    fail_count = 0
    
    for file_path in yaml_files:
        if os.path.exists(file_path):
            if convert_to_utf8(file_path):
                success_count += 1
            else:
                fail_count += 1
        else:
            logger.warning(f"File not found: {file_path}")
            fail_count += 1
    
    logger.info(f"\nYAML Encoding Fix Summary:")
    logger.info(f"  Successfully converted: {success_count}")
    logger.info(f"  Failed conversions: {fail_count}")
    logger.info(f"  Total files processed: {len(yaml_files)}")
    
    if fail_count == 0:
        logger.info("✅ All YAML files successfully converted to UTF-8!")
    else:
        logger.warning(f"⚠️  {fail_count} files had conversion issues")
    
    return fail_count == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
