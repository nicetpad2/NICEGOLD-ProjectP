#!/bin/bash
# NICEGOLD Enterprise Backup Script

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "🗄️ Creating NICEGOLD Enterprise backup..."

# Backup database
if [ -f database/production.db ]; then
    cp database/production.db "$BACKUP_DIR/"
    echo "✅ Database backed up"
fi

# Backup configuration
cp -r config "$BACKUP_DIR/"
echo "✅ Configuration backed up"

# Backup logs (last 7 days)
find logs -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/" \;
echo "✅ Recent logs backed up"

# Create archive
tar -czf "$BACKUP_DIR.tar.gz" -C backups $(basename "$BACKUP_DIR")
rm -rf "$BACKUP_DIR"

echo "✅ Backup completed: $BACKUP_DIR.tar.gz"

# Cleanup old backups (keep last 10)
cd backups
ls -t *.tar.gz | tail -n +11 | xargs -r rm
