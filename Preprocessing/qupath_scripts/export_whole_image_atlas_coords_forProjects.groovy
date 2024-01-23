import static qupath.lib.gui.scripting.QPEx.* // For intellij editor autocompletion
import static ch.epfl.biop.qupath.atlas.allen.api.AtlasTools.*

import qupath.lib.projects.ProjectIO
import qupath.lib.projects.Project
import qupath.lib.projects.Projects
import qupath.lib.images.ImageData
import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import qupath.lib.measurements.MeasurementList
import qupath.lib.objects.PathCellObject


import qupath.ext.biop.abba.AtlasTools
import qupath.ext.biop.warpy.Warpy
import ch.epfl.biop.qupath.transform.*
import net.imglib2.RealPoint

import qupath.lib.images.servers.ImageServer
import qupath.lib.regions.RegionRequest
import qupath.lib.images.servers.TileRequest


///////////////////////////////////////////////////////////////////////////////
// Description
//   This script is used to map the pixels for each image to 3d atlas coordinates
//     and stores it as .tsv files in subfolder "qupath_export_pxToAtlasCoords"
//   This will be run for all projects in the animal folders in the parent base dir
///////////////////////////////////////////////////////////////////////////////



// Define a region (could be the whole image, or a specific ROI)
int tileWidth = 8;
int tileHeight = 8;

// Define the parent directory that contains folders for each animal with qupath projects inside
def projectBaseDir = new File("H:/fullsize/fullsize");
// note to self remove "qupath/" when running cohorts 3 and 4
def projectBasePath = "qupath/project.qpproj" // moved here from down below but idk if it works

// Check if the directory exists
if (!projectBaseDir.exists()) {
    print("The specified directory does not exist!")
    return
};
if (!projectBaseDir.isDirectory()) {
    print("The specified directory is not a directory!")
    return
};


// Iterate through all the projects in the directory
projectBaseDir.eachFile { anDir ->
    ////////////
    // NOTE !!! remove "qupath/" when running cohorts 3 and 4
    ///////////
    def quProjFile = new File (anDir, projectBasePath); 
    print(quProjFile)
    if (!quProjFile.exists()) {
        print("The specified quProjFile does not exist!")
        return
    }
    
    def project = ProjectIO.loadProject(quProjFile, BufferedImage.class)
    def projectImageList = project.getImageList()

    File directory = new File(anDir, "qupath/qupath_export_pxToAtlasCoords");
    if (!directory.exists()) {
        directory.mkdir();
    };
    
    // iterate through each image in the project
    for (def entry in projectImageList) {
        // init abba funcs and args
        def targetEntryPath = entry.getEntryPath(); // gets path to qupath data for this image
        def fTransform = new File (targetEntryPath.toString(),"ABBA-Transform-Adult Mouse Brain - Allen Brain Atlas V3p1.json");
        if (!fTransform.exists()) {
            println ("ABBA transformation file not found for entry "+targetEntryPath);
            continue;
        }
        def pixelToCCFTransform = Warpy.getRealTransform(fTransform).inverse(); // Needs the inverse transform
        
        // get image data
        def imageData = entry.readImageData();
        def server = imageData.getServer();
        imageName = ServerTools.getDisplayableImageName(server);

        // prepare output file
        def filename = imageName.take(imageName.indexOf('.'));
        def outputPath = buildFilePath(directory.toString(), filename + '_pxToAtlasCoords.tsv');
        def output_file = new File(outputPath);
        // skip if output already exists
        if (output_file.exists()) {
            print (outputPath + " Already completed!");
            continue;
        }
        
        // Calculate the number of tiles in x and y direction
        def nTilesX = Math.ceil(server.getWidth() / tileWidth) +1;
        def nTilesY = Math.ceil(server.getHeight() / tileHeight) +1;

        // Iterate through the tiles
        print imageName + "w,h (" + server.getWidth()+ "," + server.getHeight() + ")  nTiles w,h:(" + nTilesX+ "," + nTilesY + ")";
        List<Map> sampledPxToAtlasCoords = []; // Store img x,y (in px) and atlas x,y,z coords in mm

        for (int y = 0; y < nTilesY; y++) {
            for (int x = 0; x < nTilesX; x++) {
                // get tile top left corner
                int centerX = x * tileWidth;
                int centerY = y * tileHeight;
                

                // get atlas coords
                RealPoint ccfCoordinates = new RealPoint(3);
                ccfCoordinates.setPosition([
                    centerX,centerY,0
                ] as double[]);
                pixelToCCFTransform.apply(ccfCoordinates, ccfCoordinates);

                // add to dict 
                sampledPxToAtlasCoords.add([x: centerX, y: centerY, ccfXmm: ccfCoordinates.getDoublePosition(0), ccfYmm: ccfCoordinates.getDoublePosition(1), ccfZmm: ccfCoordinates.getDoublePosition(2)]);

            };
        };  

        // save annotations to dir in qupath folder, Create directory 'atlas_coords' if it doesn't exist
        output_file.withPrintWriter { writer ->
            writer.println("x\ty\tccfXmm\tccfYmm\tccfZmm");  // Header
            sampledPxToAtlasCoords.each { data ->
                writer.println("${data.x}\t${data.y}\t${data.ccfXmm}\t${data.ccfYmm}\t${data.ccfZmm}")
            };
        };
        print "Done!"
    };
};