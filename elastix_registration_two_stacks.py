#
# use elastix from Fiji
#
# Author information:
# 
# takehito.tomita@embl.de
# adapted from step_2_fiji--elastix--interactive(v9).py written by Christian Tischer
# Input: 
# 
# Computation:
#
# Output:
#
#
#v2 preprocesses images (translates the images such that center of mass matches the image center, and also normalises the intensity)

import os, os.path, re, sys
from subprocess import Popen, PIPE
import time, shutil, sys, math
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
import h5py
from skimage import io
import SimpleITK as sitk
import linecache
from scipy import ndimage
import cv2
from tqdm import tqdm

def ensure_empty_dir(path):
		if os.path.isdir(path):
				print("Removing contents of "+path)
				for entry in os.listdir(path):
						abspath = os.path.join(path, entry)
						if os.path.isdir(abspath):
								shutil.rmtree(abspath)
						elif os.path.isfile(abspath):
								os.remove(abspath)
						elif os.path.lexists(abspath):
								raise Exception(
										"The path already exists. " \
										"Please remove it manually."
								)
		elif os.path.isfile(path):
				os.remove(path)
		elif os.path.lexists(path):
				raise Exception(
						"The path already exists. Please remove it manually."
				)
		else:
				print("Creating new folder "+path)
				os.mkdir(path)

def cmd(args):
	os.environ["DYLD_LIBRARY_PATH"] = "/Users/tischi/Downloads/elastix_macosx64_v4.8/lib:$DYLD_LIBRARY_PATH"
	#p = Popen(['/Users/tischi/Downloads/elastix_macosx64_v4.8/bin/elastix','--help'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
	#output, err = p.communicate()  
	p = Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE)
	output, err = p.communicate()
	return(output)

def transformix(moving_file, output_folder, p, transformation_file):
	print("  running transformix:")
	print("    transformation file: "+transformation_file)      
	print("    moving file: "+moving_file)      
	start_time = time.time()   
	output = cmd([p["transformix_binary_file"], '-in', moving_file, '-out', output_folder, '-tp',  transformation_file])
	print("    time elapsed: "+str(round(time.time()-start_time,3)))
	
	return(output)

def elastix(fixed_file, moving_file, output_folder, p, init_with_trafo = ""):
	print("  running elastix:")
	
	cmd_args = [p["elastix_binary_file"], '-priority', 'HIGH', '-f', fixed_file, '-m', moving_file, '-out', output_folder, '-p',  p["elastix_parameter_file"]]

	if init_with_trafo: 
		cmd_args.append('-t0')
		cmd_args.append(init_with_trafo)   
	 
	if p['mask_file']:
		cmd_args.append('-fMask')
		cmd_args.append(p['mask_file'])
		
	print("    initial trafo: "+init_with_trafo)
	print("    reference: "+fixed_file) 
	print("    to be transformed: "+moving_file)      
	 
	start_time = time.time()   
	output = cmd(cmd_args)
	print("    time elapsed: "+str(round(time.time()-start_time,3)))
		
	return(output)

def make_mask_file(array):
	# g_array = gaussian_filter(array,sigma = 6)
	thresholded_array = array>1010
	return thresholded_array

def read_h5file(h5_path):
	f = h5py.File(h5_path, 'r')
	ITKI = f['ITKImage']
	data = ITKI['0']
	Vdata = np.array(data['VoxelData'])

	return Vdata

def make_transformix_parameter_file(parameter_file_path,old_TP,translation_parameters,CenterOfRotationPoint_str):

	file_path = parameter_file_path
	script_file = open(file_path, "w+")

	new_TP = [old_TP[0],old_TP[1],old_TP[2],old_TP[3]+translation_parameters[3],old_TP[4]+translation_parameters[4],old_TP[5]+translation_parameters[5]]
	new_TransformParameters = str(new_TP[0])+" "+str(new_TP[1])+" "+str(new_TP[2])+" "+str(new_TP[3])+" "+str(new_TP[4])+" "+str(new_TP[5])
	
	txt = [
	'(Transform "EulerTransform")',
	'(NumberOfParameters 6)',
	'(TransformParameters '+new_TransformParameters+')',
	'(InitialTransformParametersFileName "NoInitialTransform")',
	'(HowToCombineTransforms "Compose")',
	'// Image specific',
	'(FixedImageDimension 3)',
	'(MovingImageDimension 3)',
	'(Size 512 512 120)',
	'(Index 0 0 0)',
	'(Spacing 1.0000000000 1.0000000000 1.0000000000)',
	'(Origin 0.0000000000 0.0000000000 0.0000000000)', 
	'(Direction 1.0000000000 0.0000000000 0.0000000000 0.0000000000 1.0000000000 0.0000000000 0.0000000000 0.0000000000 1.0000000000)', 
	'(UseDirectionCosines "false")', 
	'// EulerTransform specific', 
	CenterOfRotationPoint_str, 
	'(ComputeZYX "false")',
	'// ResampleInterpolator specific', 
	'(ResampleInterpolator "FinalLinearInterpolator")', 
	'// Resampler specific',
	'(Resampler "DefaultResampler")',
	'(DefaultPixelValue 1000.000000)',
	'(ResultImageFormat "h5")',
	'(ResultImagePixelType "short")', 
	'(CompressResultImage "false")']
	txt = '\n'.join(txt)
	txt = txt + '\n'
	script_file.write(txt)
	script_file.close()

	return file_path

def get_TransformParameters(transformix_parameter_file_path):
		with open(transformix_parameter_file_path, 'r') as searchfile:
				for index,line in enumerate(searchfile):
						if 'TransformParameters' in line:
								parameter_line=index

		p=linecache.getline(transformix_parameter_file_path,parameter_line)
		linecache.clearcache()

		p = p.replace('(TransformParameters ','').replace(')','')
		p = p.rstrip().split(' ')
		p = [float(i) for i in p]

		return p

def get_CenterOfRotationPoint(transformix_parameter_file_path):
		with open(transformix_parameter_file_path, 'r') as searchfile:
				for index,line in enumerate(searchfile):
						if 'CenterOfRotationPoint' in line:
								parameter_line=index+1

		p=linecache.getline(transformix_parameter_file_path,parameter_line)
		linecache.clearcache()
		return p.rstrip()

def translate_3D(moving_data,difference_vector):
		hdim,ydim,xdim = moving_data.shape

		translation_in_H = difference_vector[0]
		translation_in_Y = difference_vector[1]
		translation_in_X = difference_vector[2]

		XY_translate_matrix = np.float32([[1, 0, translation_in_X], [0, 1, translation_in_Y]])
		for h in np.arange(0,hdim):
				d = moving_data[h,:,:]
				moving_data[h,:,:] = cv2.warpAffine(d, XY_translate_matrix, d.shape[1::-1])

		H_translate_matrix = np.float32([[1, 0, 0], [0, 1, translation_in_H]])
		for y in np.arange(0,ydim):
				d = moving_data[:,y,:]
				moving_data[:,y,:] = cv2.warpAffine(d, H_translate_matrix, d.shape[1::-1])

		return moving_data

#uncomment below if running on HIVE
# elastix_binary_file = 'D:\TischiReg\elastix_windows64_v4.8\elastix.exe'
# transformix_binary_file = 'D:/TischiReg/elastix_windows64_v4.8/transformix.exe'
# elastix_parameter_file_for_Euler = 'C:/Users/acquifer/Desktop/elastix_parameter_file_templates/elastix-parameters_Euler.txt'
# elastix_parameter_file_for_translation = 'C:/Users/acquifer/Desktop/elastix_parameter_file_templates/elastix-parameters_Translation.txt'

#uncomment below if running on LightSheet
elastix_binary_file = 'E:/SWAP/Takehito/elastix_windows64_v4.8/elastix.exe'
transformix_binary_file = 'E:/SWAP/Takehito/elastix_windows64_v4.8/transformix.exe'
elastix_parameter_file_for_Euler = 'E:/SWAP/Takehito/elastix_parameter_file_templates/elastix-parameters_Euler.txt'
elastix_parameter_file_for_translation = 'E:/SWAP/Takehito/elastix_parameter_file_templates/elastix-parameters_Translation.txt'
elastix_parameter_file_for_Euler_bg_zero = 'E:/SWAP/Takehito/elastix_parameter_file_templates/elastix-parameters_Euler_bg_zero.txt'
elastix_parameter_file_for_translation_bg_zero = 'E:/SWAP/Takehito/elastix_parameter_file_templates/elastix-parameters_Translation_bg_zero.txt'


wdir = os.getcwd()
p = Path(wdir)

##########################
#parameters
x_res = 1068.18/512
y_res = 1068.18/512
z_res = 10
threshold = 230
ref_channel = 'ch1'
other_channels = ['ch0']
##########################

# output_folder = os.path.join(wdir,'transition_test')

# if not os.path.isdir(output_folder):
#     os.mkdir(output_folder)

# ensure_empty_dir(output_folder)

for sample in ['E4']:
	output_folder = os.path.join(wdir,'python_reg_'+sample)

	if not os.path.isdir(output_folder):
		os.mkdir(output_folder)

	ensure_empty_dir(output_folder)

	temporary_mhd_folder = os.path.join(wdir,'temp_mhd_'+sample)

	if not os.path.isdir(temporary_mhd_folder):
		os.mkdir(temporary_mhd_folder)

	# translated_folder =  os.path.join(output_folder,'translated_'+sample)
	# if not os.path.isdir(translated_folder):
	#     os.mkdir(translated_folder)

	first_stack_tif_names = list( p.glob('2022*_01_Average_'+sample+'_'+ref_channel+'*.tif') )
	first_stack_tif_names.sort()
	second_stack_tif_names = list( p.glob('2022*_02_Average_'+sample+'_'+ref_channel+'*.tif') )
	second_stack_tif_names.sort()

	base_name = first_stack_tif_names[0].name.replace('E7_01_Average','E7_01_to_02_Average').replace('_t000.tif','_t')

	#preprocess images
	for stack_tif_name in tqdm(first_stack_tif_names + second_stack_tif_names):
		#translate the reference channel, save as mhd
		ref_image_data = io.imread(stack_tif_name,plugin = 'tifffile')
		
		ref_image_data_COM = ndimage.measurements.center_of_mass(ref_image_data>threshold)
		difference_vector = np.array(ref_image_data.shape)/2 - np.array(ref_image_data_COM)
		translated_ref_image_data = translate_3D(ref_image_data,difference_vector)
		mhd_path = os.path.join(temporary_mhd_folder,stack_tif_name.name.replace('.tif','.mhd'))
		itk_image = sitk.GetImageFromArray(translated_ref_image_data)
		itk_image.SetSpacing([x_res,y_res,z_res])
		sitk.WriteImage(itk_image, mhd_path)

		#translate other channels,save as mhd
		for channel in other_channels:
			image_data = io.imread(os.path.join(p,stack_tif_name.name.replace(ref_channel,channel)),plugin = 'tifffile')
			_,ydim,xdim = image_data.shape
			back_ground = np.mean(image_data[:,ydim-30:ydim,0:30])
			image_data = image_data-int(back_ground)+1000
			translated_image_data = translate_3D(image_data,difference_vector)
			mhd_path = os.path.join(temporary_mhd_folder,stack_tif_name.name.replace(ref_channel,channel).replace('.tif','.mhd'))
			itk_image = sitk.GetImageFromArray(translated_image_data)
			itk_image.SetSpacing([x_res,y_res,z_res])
			sitk.WriteImage(itk_image, mhd_path)

		#process the reference channel to facilitate registration
		gauss_ref_image_data = gaussian_filter(io.imread(stack_tif_name,plugin = 'tifffile').astype(np.float32),sigma = 3)
		translated_gauss_ref_image_data = translate_3D(gauss_ref_image_data,difference_vector)
		bgr_gauss_translated_ref_image_data = translated_gauss_ref_image_data - threshold
		bgr_gauss_translated_ref_image_data[bgr_gauss_translated_ref_image_data<0] = 0
		#save as mhd
		mhd_path = os.path.join(temporary_mhd_folder,'reference_'+stack_tif_name.name.replace('.tif','.mhd'))
		itk_image = sitk.GetImageFromArray(bgr_gauss_translated_ref_image_data)
		itk_image.SetSpacing([x_res,y_res,z_res])
		sitk.WriteImage(itk_image, mhd_path)

	all_ref_mhd_names = list( Path(temporary_mhd_folder).glob('reference*.mhd') )
	all_ref_mhd_names.sort()
	reference_frame = len(first_stack_tif_names)

	reference_registration_folder = os.path.join(output_folder,'reference_registration')

	if not os.path.isdir(reference_registration_folder):
		os.mkdir(reference_registration_folder)

	#register backwards from reference frame
	fixed_mhd_path = all_ref_mhd_names[reference_frame]
	moving_mhd_path = all_ref_mhd_names[reference_frame-1]

	cmd_args = [elastix_binary_file, '-priority', 'high', '-f', str(fixed_mhd_path), '-m', str(moving_mhd_path), '-out', reference_registration_folder, '-p',  elastix_parameter_file_for_Euler_bg_zero]

	output = cmd(cmd_args)

	transformation_file = os.path.join(reference_registration_folder,'TransformParameters.0.txt')
	moved_file = os.path.join(reference_registration_folder,'result.0.h5')

	if os.path.isfile(moved_file):				
			os.rename(transformation_file,os.path.join(reference_registration_folder,'TransformParameters_' + base_name + "%03d" % (reference_frame-1) + '.txt'))
			os.rename(moved_file,os.path.join(reference_registration_folder,'transformed_reference_' + base_name + "%03d" % (reference_frame-1) + '.h5'))

	else:
			print('there was an error')

	print('registering backwards from reference frame...')
	for frame in tqdm(np.arange(1,reference_frame,1)):
		fixed_data = read_h5file(os.path.join(reference_registration_folder,'transformed_reference_' + base_name + "%03d" % (reference_frame-frame) + '.h5'))
		fixed_mhd_path = os.path.join(reference_registration_folder,'temp_fixed.mhd')
		itk_image = sitk.GetImageFromArray(fixed_data)
		itk_image.SetSpacing([x_res,y_res,z_res])
		sitk.WriteImage(itk_image, fixed_mhd_path)

		moving_mhd_path = all_ref_mhd_names[reference_frame-frame-1]

		t0 = os.path.join(reference_registration_folder,'TransformParameters_' + base_name + "%03d" % (reference_frame-frame) + '.txt')
		cmd_args = [elastix_binary_file, '-priority', 'high', '-f', str(fixed_mhd_path), '-m', str(moving_mhd_path), '-out', reference_registration_folder, '-p',  elastix_parameter_file_for_Euler_bg_zero, '-t0', t0]

		output = cmd(cmd_args)

		transformation_file = os.path.join(reference_registration_folder,'TransformParameters.0.txt')
		moved_file = os.path.join(reference_registration_folder,'result.0.h5')
		if os.path.isfile(moved_file):				
			os.rename(transformation_file,os.path.join(reference_registration_folder,'TransformParameters_' + base_name + "%03d" % (reference_frame-frame-1) + '.txt'))
			os.rename(moved_file,os.path.join(reference_registration_folder,'transformed_reference_' + base_name + "%03d" % (reference_frame-frame-1) + '.h5'))

		else:
			print('there was an error')
			break

	#register forward from the reference frame
	print('registering forward from reference frame...')
	for frame in tqdm(np.arange(reference_frame,len(all_ref_mhd_names),1)):
		fixed_data = read_h5file(os.path.join(reference_registration_folder,'transformed_reference_' + base_name + "%03d" % (frame-1) + '.h5'))
		fixed_mhd_path = os.path.join(reference_registration_folder,'temp_fixed.mhd')
		itk_image = sitk.GetImageFromArray(fixed_data)
		itk_image.SetSpacing([x_res,y_res,z_res])
		sitk.WriteImage(itk_image, fixed_mhd_path)

		moving_mhd_path = all_ref_mhd_names[frame]

		t0 = os.path.join(reference_registration_folder,'TransformParameters_' + base_name + "%03d" % (frame-1) + '.txt')
		cmd_args = [elastix_binary_file, '-priority', 'high', '-f', str(fixed_mhd_path), '-m', str(moving_mhd_path), '-out', reference_registration_folder, '-p',  elastix_parameter_file_for_Euler_bg_zero, '-t0', t0]

		output = cmd(cmd_args)

		transformation_file = os.path.join(reference_registration_folder,'TransformParameters.0.txt')
		moved_file = os.path.join(reference_registration_folder,'result.0.h5')
		if os.path.isfile(moved_file):				
			os.rename(transformation_file,os.path.join(reference_registration_folder,'TransformParameters_' + base_name + "%03d" % (frame) + '.txt'))
			os.rename(moved_file,os.path.join(reference_registration_folder,'transformed_reference_' + base_name + "%03d" % (frame) + '.h5'))

		else:
			print('there was an error')
			break

	#Transform the original mhds based on transformation parameters obtained
	channels = other_channels
	channels.append(ref_channel)

	for channel in channels:
			channel_mhd_names = list( Path(temporary_mhd_folder).glob('*'+channel+'*.mhd') )
			channel_mhd_names = [i for i in channel_mhd_names if 'reference' not in i.name]
			channel_mhd_names.sort()

			print('transforming ' + channel)
			for frame in tqdm(np.arange(len(channel_mhd_names))):
				transformation_file = os.path.join(reference_registration_folder,'TransformParameters_' + base_name + "%03d" % (frame) + '.txt')
				if os.path.isfile(transformation_file):
					output = cmd([transformix_binary_file, '-in', str(channel_mhd_names[frame]), '-out', output_folder, '-tp',  transformation_file ])
					os.rename(os.path.join(output_folder,'result.h5'),os.path.join(output_folder,'transformed_' + base_name.replace(ref_channel,channel) + "%03d" % (frame) + '.h5'))



		# base_name = first_stack_tif_names[0].name.replace('E7_01_Average','E7_01_to_02_Average').replace('_t000.tif','_t')

		# first_stack_frame_num = len(first_stack_tif_names)

		# #the first frame of the second stack
		# fixed_data = io.imread(second_stack_tif_names[0],plugin = 'tifffile')
		# #the last frame of the second stack
		# moving_data = io.imread(first_stack_tif_names[-1],plugin = 'tifffile')

		# #Obtain center of mass for rough translation
		# fixed_data_COM = ndimage.measurements.center_of_mass(fixed_data>threshold)
		# moving_data_COM = ndimage.measurements.center_of_mass(moving_data>threshold)

		# difference_vector = np.array(fixed_data_COM) - np.array(moving_data_COM)

		# moving_data = translate_3D(moving_data,difference_vector)

		# gauss_moving_data = gaussian_filter(moving_data,sigma = 3)
		# gauss_fixed_data = gaussian_filter(fixed_data,sigma = 3)

		# gauss_moving_data[gauss_moving_data<230] = 0
		# gauss_fixed_data[gauss_fixed_data<230] = 0

		# io.imsave(os.path.join(output_folder,'translated.tif'),moving_data.astype(np.float32))

		# #make mhd files with the appropriate XYZ scale
		# fixed_itk_image = sitk.GetImageFromArray( gauss_fixed_data)
		# fixed_itk_image.SetSpacing([x_res,y_res,z_res]) # Each pixel is 1.1 x 0.98 mm^2
		# sitk.WriteImage(fixed_itk_image, os.path.join(output_folder,'temp_fixed.mhd'))
		# moving_itk_image = sitk.GetImageFromArray( gauss_moving_data)
		# moving_itk_image.SetSpacing([x_res,y_res,z_res]) # Each pixel is 1.1 x 0.98 mm^2
		# sitk.WriteImage(moving_itk_image, os.path.join(output_folder,'temp_moving.mhd'))
		# fixed_mhd_path = os.path.join(output_folder,'temp_fixed.mhd')
		# moving_mhd_path = os.path.join(output_folder,'temp_moving.mhd')

		# cmd_args = [elastix_binary_file, '-priority', 'high', '-f', str(fixed_mhd_path), '-m', str(moving_mhd_path), '-out', output_folder, '-p',  elastix_parameter_file_for_translation_bg_zero]

		# output = cmd(cmd_args)

		# transformation_file = os.path.join(output_folder,'TransformParameters.0.txt')
		# moved_file = os.path.join(output_folder,'result.0.h5')
		# frame = len(first_stack_tif_names)

		# if os.path.isfile(moved_file):
				

		# 		# os.rename(moved_file,os.path.join(output_folder,'transformed_' + base_name + "%03d" % (frame-1) + '.h5'))
		# 		os.rename(transformation_file,os.path.join(output_folder,'Translation_parameters_between_two_stacks.txt'))
		# 		#move the original data with obtained parameters
		# 		moving_itk_image = sitk.GetImageFromArray(moving_data)
		# 		moving_itk_image.SetSpacing([x_res,y_res,z_res]) # Each pixel is 1.1 x 0.98 mm^2
		# 		sitk.WriteImage(moving_itk_image, os.path.join(output_folder,'temp_moving.mhd'))

		# 		transformation_file_path = os.path.join(output_folder,'Translation_parameters_between_two_stacks.txt')
		# 		output = cmd([transformix_binary_file, '-in', str(os.path.join(output_folder,'temp_moving.mhd')), '-out', output_folder, '-tp',  transformation_file_path ])
		# 		os.rename(os.path.join(output_folder,'result.h5'),os.path.join(output_folder,'transformed_' + base_name + "%03d" % (frame-1) + '.h5'))


		# else:
		# 		print('there was an error at '+str(frame))
		# 		break

		# #transform ch0
		# input_data_path = os.path.join(p,first_stack_tif_names[-1].name.replace('ch1','ch0'))
		# input_data = io.imread(input_data_path,plugin = 'tifffile')
		# input_data = translate_3D(input_data,difference_vector)
		# #first make temporary mhd
		# input_data_itk_image = sitk.GetImageFromArray(input_data)
		# input_data_itk_image.SetSpacing([x_res,y_res,z_res]) # Each pixel is 1.1 x 0.98 mm^2
		# sitk.WriteImage(input_data_itk_image, os.path.join(output_folder,'input_data.mhd'))

		# transformation_file_path = os.path.join(output_folder,'Translation_parameters_between_two_stacks.txt')
		# output = cmd([transformix_binary_file, '-in', str(os.path.join(output_folder,'input_data.mhd')), '-out', output_folder, '-tp',  transformation_file_path ])
		# os.rename(os.path.join(output_folder,'result.h5'),os.path.join(output_folder,'transformed_' + base_name.replace('ch1','ch0') + "%03d" % (frame-1) + '.h5'))



		# # convert h5 to tif file for easy check
		# # moved_data = read_h5file(os.path.join(output_folder,'transformed_' + base_name + "%03d" % (frame-1) + '.h5'))
		# # io.imsave(os.path.join(output_folder,'transformed_' + base_name + "%03d" % (frame-1) + '.tif'),moved_data.astype(np.float32))

		# #register the first stack backwards
		# # for frame in np.arange(len(first_stack_tif_names)-1,0,-1):
		# for frame in np.arange(len(first_stack_tif_names)-1,90,-1):
		# 		fixed_file = os.path.join(output_folder,'transformed_' + base_name + "%03d" % frame + '.h5')
		# 		moving_file = first_stack_tif_names[frame-1]

		# 		fixed_file_data = read_h5file(fixed_file)
		# 		gauss_fixed_data = gaussian_filter(fixed_file_data,sigma = 3)
		# 		gauss_fixed_data[gauss_fixed_data<230] = 0

		# 		moving_file_data = io.imread(moving_file,plugin = 'tifffile')
		# 		moving_file_data = translate_3D(moving_file_data,difference_vector)

		# 		#first make temporary mhds
		# 		fixed_itk_image = sitk.GetImageFromArray( gauss_fixed_data)
		# 		fixed_itk_image.SetSpacing([x_res,y_res,z_res]) # Each pixel is 1.1 x 0.98 mm^2
		# 		sitk.WriteImage(fixed_itk_image, os.path.join(output_folder,'temp_fixed.mhd'))
		# 		moving_itk_image = sitk.GetImageFromArray( moving_file_data)
		# 		moving_itk_image.SetSpacing([x_res,y_res,z_res]) # Each pixel is 1.1 x 0.98 mm^2
		# 		sitk.WriteImage(moving_itk_image, os.path.join(output_folder,'temp_moving.mhd'))

		# 		#translate the moving file based on the previous result before running registration
		# 		translation_parameter_txt_path = os.path.join(output_folder,'Translation_parameters_between_two_stacks.txt')

		# 		output = cmd([transformix_binary_file, '-in', str(os.path.join(output_folder,'temp_moving.mhd')), '-out', output_folder, '-tp',  translation_parameter_txt_path ])
		# 		translated_moving_file_data = read_h5file(os.path.join(output_folder,'result.h5'))

		# 		#reinitialize the moving data as mhd
		# 		gauss_moving_data = gaussian_filter(translated_moving_file_data,sigma = 3)
		# 		gauss_moving_data[gauss_moving_data<230] = 0
		# 		translated_moving_itk_image = sitk.GetImageFromArray(gauss_moving_data)
		# 		translated_moving_itk_image.SetSpacing([x_res,y_res,z_res]) # Each pixel is 1.1 x 0.98 mm^2
		# 		sitk.WriteImage(translated_moving_itk_image, os.path.join(output_folder,'temp_translated_moving.mhd'))

		# 		#run the registration between the two files
		# 		fixed_mhd_path = os.path.join(output_folder,'temp_fixed.mhd')
		# 		translated_moving_mhd_path = os.path.join(output_folder,'temp_translated_moving.mhd')

		# 		t0 = os.path.join(output_folder,'TransformParameters_' + base_name + "%03d" % frame + '.txt')
		# 		if os.path.isfile(t0):
		# 				cmd_args = [elastix_binary_file, '-priority', 'high', '-f', str(fixed_mhd_path), '-m', str(translated_moving_mhd_path), '-out', output_folder, '-p',  elastix_parameter_file_for_Euler, '-t0', t0]
		# 		else:
		# 				cmd_args = [elastix_binary_file, '-priority', 'high', '-f', str(fixed_mhd_path), '-m', str(translated_moving_mhd_path), '-out', output_folder, '-p',  elastix_parameter_file_for_Euler]

		# 		output = cmd(cmd_args)

		# 		transformation_file = os.path.join(output_folder,'TransformParameters.0.txt')
		# 		moved_file = os.path.join(output_folder,'result.0.h5')

		# 		if os.path.isfile(moved_file):

		# 				os.rename(moved_file,os.path.join(output_folder,'transformed_' + base_name + "%03d" % (frame-1) + '.h5'))
		# 				os.rename(transformation_file,os.path.join(output_folder,'TransformParameters_' + base_name + "%03d" % (frame-1) + '.txt'))

		# 				#transform ch0
		# 				input_data_path = os.path.join(p,first_stack_tif_names[frame-1].name.replace('ch1','ch0'))
		# 				input_data = io.imread(input_data_path,plugin = 'tifffile')
		# 				#first make temporary mhd
		# 				input_data_itk_image = sitk.GetImageFromArray(input_data)
		# 				input_data_itk_image.SetSpacing([x_res,y_res,z_res]) # Each pixel is 1.1 x 0.98 mm^2
		# 				sitk.WriteImage(input_data_itk_image, os.path.join(output_folder,'input_data.mhd'))
		# 				#first translate
		# 				transformation_file_path = os.path.join(output_folder,'Translation_parameters_between_two_stacks.txt')
		# 				output = cmd([transformix_binary_file, '-in', str(os.path.join(output_folder,'input_data.mhd')), '-out', output_folder, '-tp',  transformation_file_path ])

		# 				translated_output = read_h5file(os.path.join(output_folder,'result.h5'))

		# 				#reinitialize the moving data as mhd
		# 				translated_output_itk_image = sitk.GetImageFromArray(translated_output)
		# 				translated_output_itk_image.SetSpacing([x_res,y_res,z_res]) # Each pixel is 1.1 x 0.98 mm^2
		# 				sitk.WriteImage(translated_output_itk_image, os.path.join(output_folder,'temp_translated_output.mhd'))

		# 				#transform with Euler parameters
		# 				transformation_file_path = os.path.join(output_folder,'TransformParameters_' + base_name + "%03d" % (frame-1) + '.txt')
		# 				output = cmd([transformix_binary_file, '-in', str(os.path.join(output_folder,'temp_translated_output.mhd')), '-out', output_folder, '-tp',  transformation_file_path ])
		# 				os.rename(os.path.join(output_folder,'result.h5'),os.path.join(output_folder,'transformed_' + base_name.replace('ch1','ch0') + "%03d" % (frame-1) + '.h5'))


		# 		else:
		# 				print('there was an error at '+str(frame))
		# 				break

		# #register the second stack onwards
		# # for frame in np.arange(0,len(second_stack_tif_names),1):
		# for frame in np.arange(0,5,1):

		# # for frame in np.arange(1,3,1):
		#     fixed_file = os.path.join(output_folder,'transformed_' + base_name + "%03d" % (first_stack_frame_num+frame-1) + '.h5')
				
		#     moving_file = second_stack_tif_names[frame]
		#     # output = cmd([transformix_binary_file, '-in', str(moving_file), '-out', output_folder, '-tp',  stack_translation_file])
		#     # trans_moving_file = os.path.join(output_folder,'result.h5')

		#     fixed_file_data = read_h5file(fixed_file)
		#     moving_file_data = io.imread(moving_file,plugin = 'tifffile')
		#     fixed_itk_image = sitk.GetImageFromArray( fixed_file_data)
		#     fixed_itk_image.SetSpacing([x_res,y_res,z_res]) # Each pixel is 1.1 x 0.98 mm^2
		#     sitk.WriteImage(fixed_itk_image, os.path.join(output_folder,'temp_fixed.mhd'))
		#     moving_itk_image = sitk.GetImageFromArray( moving_file_data)
		#     moving_itk_image.SetSpacing([x_res,y_res,z_res]) # Each pixel is 1.1 x 0.98 mm^2
		#     sitk.WriteImage(moving_itk_image, os.path.join(output_folder,'temp_moving.mhd'))
		#     fixed_mhd_path = os.path.join(output_folder,'temp_fixed.mhd')
		#     moving_mhd_path = os.path.join(output_folder,'temp_moving.mhd')

		#     t0 = os.path.join(output_folder,'TransformParameters_' + base_name + "%03d" % (first_stack_frame_num+frame-1) + '.txt')
		#     if os.path.isfile(t0):
		#         cmd_args = [elastix_binary_file, '-priority', 'high', '-f', str(fixed_mhd_path), '-m', str(moving_mhd_path), '-out', output_folder, '-p',  elastix_parameter_file_for_Euler, '-t0', t0]
		#     else:
		#         cmd_args = [elastix_binary_file, '-priority', 'high', '-f', str(fixed_mhd_path), '-m', str(moving_mhd_path), '-out', output_folder, '-p',  elastix_parameter_file_for_Euler]

		#     output = cmd(cmd_args)

		#     transformation_file = os.path.join(output_folder,'TransformParameters.0.txt')
		#     moved_file = os.path.join(output_folder,'result.0.h5')

		#     if os.path.isfile(moved_file):

		#         os.rename(moved_file,os.path.join(output_folder,'transformed_' + base_name + "%03d" % (first_stack_frame_num+frame) + '.h5'))
		#         os.rename(transformation_file,os.path.join(output_folder,'TransformParameters_' + base_name + "%03d" % (first_stack_frame_num+frame) + '.txt'))


		#         #transform ch0
		#         input_data_path = os.path.join(p,second_stack_tif_names[frame].name.replace('ch1','ch0'))
		#         input_data = io.imread(input_data_path,plugin = 'tifffile')
		#         #first make temporary mhd
		#         input_data_itk_image = sitk.GetImageFromArray(input_data)
		#         input_data_itk_image.SetSpacing([x_res,y_res,z_res]) # Each pixel is 1.1 x 0.98 mm^2
		#         sitk.WriteImage(input_data_itk_image, os.path.join(output_folder,'input_data.mhd'))

		#         #transform with Euler parameters
		#         transformation_file_path = os.path.join(output_folder,'TransformParameters_' + base_name + "%03d" % (first_stack_frame_num+frame) + '.txt')
		#         output = cmd([transformix_binary_file, '-in', str(os.path.join(output_folder,'input_data.mhd')), '-out', output_folder, '-tp',  transformation_file_path ])
		#         os.rename(os.path.join(output_folder,'result.h5'),os.path.join(output_folder,'transformed_' + base_name.replace('ch1','ch0') + "%03d" % (first_stack_frame_num+frame) + '.h5'))



		#     else:
		#         print('there was an error at '+str(frame))
		#         break

		# if os.path.isfile(os.path.join(output_folder,'result.0.h5')):
		#     os.remove(os.path.join(output_folder,'result.0.h5'))

		# if os.path.isfile(os.path.join(output_folder,'result.h5')):
		#     os.remove(os.path.join(output_folder,'result.h5'))
