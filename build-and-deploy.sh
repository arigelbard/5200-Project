###############################################################
### Optionally rerun python scripts
read -p "Rerun Python scripts before building? (y/n): " rerun

if [ "$rerun" = "y" ]; then
    echo ""
    echo "Running scripts/process_corn.py..."
    python scripts/process_corn.py
    if [ $? -ne 0 ]; then
        echo "ERROR: process_corn.py failed. Aborting build."
        exit 1
    fi

    echo ""
    echo "Running scripts/process_cdl.py..."
    python scripts/process_cdl.py
    if [ $? -ne 0 ]; then
        echo "ERROR: process_cdl.py failed. Aborting build."
        exit 1
    fi

    echo ""
    echo "Running scripts/ghg_lifecycle_bar.py..."
    python scripts/ghg_lifecycle_bar.py
    if [ $? -ne 0 ]; then
        echo "ERROR: ghg_lifecycle_bar.py failed. Aborting build."
        exit 1
    fi

    echo ""
    echo "All scripts completed successfully."
fi

###############################################################
### Clean up step
rm -rf _site 
mkdir _site 

#################################################################
### Copy relevant files into folder
# Copy HTML pages
cp outputs/index.html _site/
cp scripts/corn_acreage_map/corn_acreage_map.html _site/
cp scripts/corn_belt_map/corn_belt_map.html _site/

# Copy static assets
cp outputs/carbon_emissions_bar.png _site/

# Copy GeoJSON data files
cp outputs/corn_acreage_change.geojson _site/
cp outputs/corn_belt_2006.geojson _site/
cp outputs/corn_belt_2007.geojson _site/
cp outputs/corn_belt_2008.geojson _site/
cp outputs/corn_belt_2009.geojson _site/
cp outputs/corn_belt_2010.geojson _site/
cp outputs/corn_belt_2011.geojson _site/
cp outputs/corn_belt_2012.geojson _site/

#################################################################
### Set folder permissions
folders=$(find _site/* -type d)
for f in $folders; do 
    chmod 755 "$f"
done 

### Set file permissions
files=$(find _site/* -type f)
for f in $files; do 
    chmod 644 "$f"
done

####################################################################
# Prompt user to push to website
printf 'Do you want to push the website to the Georgetown University domains folder? (y/n)'
read answer 

if [ "$answer" != "${answer#[Yy]}" ] ;then 
    rsync -Prltvc --delete _site/* arigelba@gtown03.reclaimhosting.com:/home/arigelba/public_html/dsan5200-project
else 
    echo NOT PUSHING TO WEBSITE!
fi