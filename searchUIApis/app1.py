import json
import zipfile
from flask import Flask, Response, request, send_file, send_from_directory
from searchResultStore import SearchResultStore
import tifffile as tiff
import io
from storeSearch import TiffImageCacheSearch
from PIL import Image,ImageColor, ImageOps
from gevent.pywsgi import WSGIServer
import numpy as np

app = Flask(__name__)

# Replace 'your_tiff_image_path' with the actual path to your TIFF image file
# tiff_image_path = './HBM233.GTZN.466/reg001_expr.ome.tif'

def color_code_to_rgb_16bit(hex_color):
    # Remove the '#' symbol if it exists
    hex_color = hex_color.lstrip('#')
    
    # Convert the hex color to decimal values
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    
    return int((red/255)*65536), int((green/255)*65536) , int((blue/255)*65536) 
    # return (red,green,blue)
    
def getpageIndexFromChannelName(channelName):
    storeObject = TiffImageCacheSearch()
    channelnames = storeObject.get_channel_names()
    return channelnames.index(channelName)



@app.route('/get_reshaped_tiff_page/<int:page>')
def get_reshaped_tiff_page(page):
    try:
        # Read the TIFF image using tiff.imread
        store_object = TiffImageCacheSearch()
        tif_image = store_object.load_tiff_image()
        a,b,width,length=tif_image.shape
        tif_image=tif_image.reshape(a*b,width,length)
        print(tif_image.shape)
        print(store_object.get_imagePath())
        if page >= len(tif_image):
            return "Page not found", 404

        # Get the specified page
        
        page_data = tif_image[page]
        
        # Reshape the page to 100x100 pixels using PIL
        reshaped_image = Image.fromarray(page_data).resize((250, 250))

        # Convert the reshaped image to bytes
        output_buffer = io.BytesIO()
        reshaped_image.save(output_buffer, format='PNG')
        # reshaped_image.save("./out.png", format='PNG')
        # Return the byte data as a Flask Response
        return Response(
            output_buffer.getvalue(),
            mimetype='image/png',
            headers={'Content-Disposition': f'attachment; filename=page_{page}_reshaped.png'}
        )

        # return send_file(
        #     io.BytesIO(output_buffer.getvalue()),
        #     mimetype='image/tiff',
        #     as_attachment=True,
        #     download_name=f'page_{page}_reshaped.tiff'
        # )
    except Exception as e:
        print(e)
        return str(e), 404
    
@app.route('/get_colored_tiff_page/<string:channelname>/<colorCode>/<int:minPixel>')
def get_colored_tiff_page(channelname, colorCode, minPixel=0):
    # Read the TIFF image using tiff.imread
    page = getpageIndexFromChannelName(channelname)
    print("page index : ",page,", channelName : ",channelname)
    store_object = TiffImageCacheSearch()
    tif_image = store_object.load_tiff_image()
    a,b,width,length=tif_image.shape
    tif_image=tif_image.reshape(a*b,width,length)
    print(tif_image.shape)
    if page >= len(tif_image):
        return "Page not found", 404
    page_data = tif_image[page]
    # output_buffer = colorize_Page_given_color(page_data, colorCode)
    if(',' in colorCode):
        k = colorCode.split(',')
        RGBColorData = (k[0],k[1],k[3])
    else:
        RGBColorData = color_code_to_rgb_16bit(colorCode)
    # page_array = page_data.asarray()
    page_array = Image.fromarray(page_data).resize((1080, 1080))
    newImage = Image.new("RGBA", page_array.size).resize((1080, 1080))
    for y in range(page_array.height):
        for x in range(page_array.width):
            pixel = page_array.getpixel((x, y))
            if pixel > minPixel:
                pixel = float((pixel / 65536))
                present_color = RGBColorData
                newImage.putpixel((x,y),tuple([int((pixel*present_color[0])**(1/2)),int((pixel*present_color[1])**(1/2)),int((pixel*present_color[2])**(1/2)),65536]))
            else:
                newImage.putpixel((x,y),(0,0,0,0))
    output_buffer = io.BytesIO()
    newImage.save(output_buffer, format='PNG')

    # Get the specified page
    
    # Return the byte data as a Flask Response
    return Response(
        output_buffer.getvalue(),
        mimetype='image/png',
        headers={'Content-Disposition': f'attachment; filename=page_{channelname}_{colorCode}_{minPixel}.png'}
    )
    # return send_file(
    #     io.BytesIO(output_buffer.getvalue()),
    #     mimetype='image/png',
    #     as_attachment=True,
    #     download_name=f'page_{channelname}_{colorCode}.png'
    # )


  
@app.route('/preprocssed_tiff_page/<colorCode>/<int:minPixel>')
def preprocssed_tiff_page(colorCode, minPixel=0):
    # Read the TIFF image using tiff.imread
    storeObject = TiffImageCacheSearch()
    channelnames = storeObject.get_channel_names()
    for i in range(len(channelnames)):
        channelname = channelnames[i]
        page = getpageIndexFromChannelName(channelname)
        print("page index : ",page,", channelName : ",channelname)
        store_object = TiffImageCacheSearch()
        tif_image = store_object.load_tiff_image()
        print(tif_image.shape)
        if page >= len(tif_image):
            return "Page not found", 404
        a,b,width,length=tif_image.shape
        tif_image=tif_image.reshape(a*b,width,length)
        page_data = tif_image[page]
        # output_buffer = colorize_Page_given_color(page_data, colorCode)
        if(',' in colorCode):
            k = colorCode.split(',')
            RGBColorData = (k[0],k[1],k[3])
        else:
            RGBColorData = color_code_to_rgb_16bit(colorCode)
        # page_array = page_data.asarray()
        page_array = Image.fromarray(page_data).resize((4500, 4500))
        newImage = Image.new("RGBA", page_array.size).resize((4500, 4500))
        for y in range(page_array.height):
            for x in range(page_array.width):
                pixel = page_array.getpixel((x, y))
                if pixel > minPixel:
                    pixel = float((pixel / 65536))
                    present_color = RGBColorData
                    newImage.putpixel((x,y),tuple([int((pixel*present_color[0])**(1/2)),int((pixel*present_color[1])**(1/2)),int((pixel*present_color[2])**(1/2)),65536]))
                else:
                    newImage.putpixel((x,y),(0,0,0,0))
        output_buffer = io.BytesIO()
        newImage.save(output_buffer, format='PNG')

        # Get the specified page

        # Return the byte data as a Flask Response
        yield Response(
            output_buffer.getvalue(),
            mimetype='image/png',
            headers={'Content-Disposition': f'attachment; filename=page_{channelname}_{colorCode}_{minPixel}.png'}
        )

@app.route('/get_channel_names/')
def get_channel_names():
    storeObject = TiffImageCacheSearch()
    list_channel_names = storeObject.get_channel_names()
    print(storeObject.get_imagePath())
    if len(list_channel_names) > 0:
        return app.response_class(response=json.dumps({"channel_names": list_channel_names}),status=200,mimetype='application/json')
    else:
        return app.response_class(response=json.dumps({"channel_names": list_channel_names}),status=500,mimetype='application/json')
    
@app.route('/set_filepath/<string:path>')
def set_filepath(path):
    storeObject = TiffImageCacheSearch()
    list_channel_names = storeObject.set_imagePath(path=path)
    return app.response_class(status=200)


@app.route('/getsearchresults/')
def getsearchresults():
    path = r"/Users/guttikondaparthasai/Downloads/pathology_slide_detection/search_engine_algorithm/results.txt"
    res = []
    with open(path, 'r') as f:
        res = f.readlines()
    return app.response_class(response=json.dumps({"result": res}),status=200,mimetype='application/json')

@app.route('/setsearchresults/')
def setSearchResults():
    data = request.json
    k = SearchResultStore()
    k.set_imagePath(list(data))
    return app.response_class(status=200)

@app.route('/getsearchinput/')
def getSearchInput():
    image_path = f"/Users/guttikondaparthasai/Downloads/pathology_slide_detection/search_engine_algorithm/cropped_1/cropped_0.png"

    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()

    return Response(
            image_data,
            mimetype='image/png',
            headers={'Content-Disposition': f'attachment; filename=image.png'}
        )
    # return Response(image_data, content_type='image/png')
    # return send_from_directory('images', "/Users/guttikondaparthasai/Downloads/pathology_slide_detection/search_engine_algorithm/cropped_1/cropped_0.png", as_attachment=True, mimetype='image/png')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
    # get_reshaped_tiff_page(0)