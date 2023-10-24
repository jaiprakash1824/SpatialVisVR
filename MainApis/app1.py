import json
import zipfile
from flask import Flask, Response, request, send_file
import tifffile as tiff
import io
from store import TiffImageCache
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
    storeObject = TiffImageCache()
    channelnames = storeObject.get_channel_names()
    return channelnames.index(channelName)



@app.route('/get_reshaped_tiff_page/<int:page>')
def get_reshaped_tiff_page(page):
    try:
        # Read the TIFF image using tiff.imread
        store_object = TiffImageCache()
        tif_image = store_object.load_tiff_image()
        print(tif_image.shape)
        if page >= len(tif_image):
            return "Page not found", 404

        # Get the specified page
        page_data = tif_image[page]

        # Reshape the page to 100x100 pixels using PIL
        reshaped_image = Image.fromarray(page_data).resize((250, 250))

        # Convert the reshaped image to bytes
        output_buffer = io.BytesIO()
        reshaped_image.save(output_buffer, format='PNG')

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
        return str(e), 404
    
@app.route('/get_colored_tiff_page/<string:channelname>/<colorCode>/<int:minPixel>')
def get_colored_tiff_page(channelname, colorCode, minPixel=0):
    # Read the TIFF image using tiff.imread
    page = getpageIndexFromChannelName(channelname)
    print("page index : ",page,", channelName : ",channelname)
    store_object = TiffImageCache()
    tif_image = store_object.load_tiff_image()
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
    storeObject = TiffImageCache()
    channelnames = storeObject.get_channel_names()
    for i in range(len(channelnames)):
        channelname = channelnames[i]
        page = getpageIndexFromChannelName(channelname)
        print("page index : ",page,", channelName : ",channelname)
        store_object = TiffImageCache()
        tif_image = store_object.load_tiff_image()
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
    storeObject = TiffImageCache()
    list_channel_names = storeObject.get_channel_names()
    print(list_channel_names)
    return app.response_class(response=json.dumps({"channel_names": list_channel_names}),status=200,mimetype='application/json')

    # if len(list_channel_names) > 0:
    #     return app.response_class(response=json.dumps({"channel_names": list_channel_names}),status=200,mimetype='application/json')
    # else:
    #     return app.response_class(response=json.dumps({"channel_names": list_channel_names}),status=500,mimetype='application/json')

@app.route('/set_filepath/<string:path>')
def set_filepath(path):
    storeObject = TiffImageCache()
    storeObject.set_imagePath(path=path)
    print(storeObject.get_imagePath())
    return app.response_class(status=200)

@app.route('/get_all_files/')
def getAllfiles():
    storeObject = TiffImageCache()
    k = storeObject.get_all_file_names()
    return app.response_class(response=json.dumps({"result": k}),status=200,mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
