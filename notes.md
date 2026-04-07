# Random Notes and Thoughts

## Land Use section
### Corn Acreage vs Natural Land Maps
So the corn acreage and grassland maps look a bit different. It seems like, according to this data, there was an increase in corn acreage planted in the dakotas, but not as much in Illinois or Indiana. Meanwhile, the CDL maps show a lot of grassland and wetland being converted to corn in Illinois and Indiana. At first, I thought this might have been a mistake or gone against our narrative, but these two maps are showing different things. The likely story is that in the Dakotas, existing farmland transitioned to corn while in Illinois and Indiana, natural land was turned to corn. 

### Interactive HTML
I also created an html file for an interactive natural land conversion map. To run it, you need to run `python -m http.server 8000` from the folder where corn_belt_map.html is and then click on it. Also note that `corn_belt_conversion.geojson` needs to be in the same folder as the html file. 

### Next steps
I think next, I want to make a map with a slider that will show the evolution of land use over time. 

## Sources
- Corn acreage data - USDA NASS Quick Stats: https://quickstats.nass.usda.gov/
- Natural Land conversion data - NASS CropScape: https://cat.csiss.gmu.edu/CropSmart/signin?redirecturl=https://nassgeodata.gmu.edu/CropScape 