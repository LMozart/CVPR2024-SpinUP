import Imath
import numpy as np
from PIL import Image
from OpenEXR import InputFile, OutputFile, Header

def read_exr(filename, channel=3):
	"""Reads an OpenEXR file and returns the data as a numpy array.
	Args:
	    filename: path to exr file
		channel: number of channels in exr file
	Return
		rgb32f: numpy array of shape [height, width, channel]
	"""
	src = InputFile(filename)
	pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
	dw = src.header()['dataWindow']
	size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

	rgb32f = None
	if channel == 3:
		for i, c in enumerate('RGB'):
			c32f = np.fromstring(src.channel(c, pixel_type), dtype=np.float32).reshape(size[::-1])
			rgb32f = c32f if i == 0 else np.dstack((rgb32f, c32f))
	elif channel == 1:
		rgb32f = np.fromstring(src.channel('A', pixel_type), dtype=np.float32).reshape(size[::-1]).unsqueeze(2)

	return rgb32f


def write_exr(filename, data):
	"""Write an OpenEXR file from a numpy array.
	Args:
		filename: path to exr file
		data: numpy array to write to exr file
	"""
	assert '.exr' in filename, 'extension must be .exr'
	assert data.dtype == np.float32, f'Data type is {type(data)}, should be np.float32'

	out = OutputFile(filename, Header(data.shape[1], data.shape[0]))

	if data.shape[2] == 3:
		r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]
	elif data.shape[2] == 1:
		r = g = b = data[:, :, 0]
	out.writePixels({'R': r.tostring(), 'G': g.tostring(), 'B': b.tostring()})

	out.close()
