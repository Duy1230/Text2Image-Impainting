import { useState, useRef, useEffect } from 'react'
import axiosInstance from '../axiosInstance'

export default function Component() {
  const [imageSegment, setImageSegment] = useState(null)
  const [imageInpainting, setImageInpainting] = useState(null)
  const [coordinates, setCoordinates] = useState(null)
  const [apiResponse, setApiResponse] = useState(null)
  const [singleTargetMode, setSingleTargetMode] = useState(true)
  const [postprocessMode, setPostprocessMode] = useState(false)
  const [isApplyingBlur, setIsApplyingBlur] = useState(true)
  const [usingCannyControl, setUsingCannyControl] = useState(true)
  const [numInferenceSteps, setNumInferenceSteps] = useState(12)
  const [guidanceScale, setGuidanceScale] = useState(7.5)
  const [controlnetScale, setControlnetScale] = useState(0.5)
  const [numSamples, setNumSamples] = useState(1)
  const imageMaskRef = useRef(null)
  const imageInpaintingRef = useRef(null)
  const segmentPrompt = useRef(null)
  const inpaintingPrompt = useRef(null)

  useEffect(() => {
    // Cleanup the object URL when the component unmounts or image changes
    return () => {
      if (imageSegment) {
        URL.revokeObjectURL(imageSegment)
      }
      if (imageInpainting) {
        URL.revokeObjectURL(imageInpainting)
      }
    }
  }, [imageSegment, imageInpainting])

  const handleImageClick = async (event) => {
    if (imageMaskRef.current) {
      const rect = imageMaskRef.current.getBoundingClientRect()
      const normalizedX = Math.round(event.clientX - rect.left) / rect.width
      const normalizedY = Math.round(event.clientY - rect.top) / rect.height


      setCoordinates({ x: normalizedX, y: normalizedY })
      console.log("clicked coordinates: ", normalizedX, normalizedY)
      try {
        // Update the field names to match the backend SegmentRequest model
        const response = await axiosInstance.post('/sam2/segment', {
          normalized_x: normalizedX,
          normalized_y: normalizedY
        }, { responseType: 'blob' })
        
        // Create a URL from the Blob
        const imageUrl = URL.createObjectURL(response.data)
        
        // Update the image state with the new URL
        setImageSegment(imageUrl)
      } catch (error) {
        console.error('Error segmenting image:', error)
        setApiResponse('Error segmenting image')
      }
    }
  }

  const handleTextSegment = async () => {
    if (segmentPrompt.current.value === "") {
      setApiResponse("Please enter a text prompt")
      //change border to red
      segmentPrompt.current.style.borderColor = "red"
      segmentPrompt.current.style.borderWidth = "2px"
      segmentPrompt.current.placeholder = "Please enter a text prompt"
    }
    else if (inpaintingPrompt.current.value === "") {
      setApiResponse("Please enter an inpainting prompt")
      inpaintingPrompt.current.style.borderColor = "red"
      inpaintingPrompt.current.style.borderWidth = "2px"
      inpaintingPrompt.current.placeholder = "Please enter an inpainting prompt"
    }
    else {
      try {
        // First get the boxes from GroundingDINO
        const dinoResponse = await axiosInstance.post('/groundingdino/predict', { 
          prompt: segmentPrompt.current.value,
          single_target_mode: singleTargetMode
        })
        
        // Then use the boxes for SAM2 segmentation
        const response = await axiosInstance.post('/sam2/segment_with_text', {
          boxes: dinoResponse.data.boxes,
        }, { responseType: 'blob' })

        // Create a URL from the Blob
        const imageSegmentUrl = URL.createObjectURL(response.data)
        setImageSegment(imageSegmentUrl)

        // Get masks from SAM2 (shape must be (1, H, W))
        const masksResponse = await axiosInstance.get('/sam2/get_masks')
        console.log(masksResponse.data)
        
        // Inpainting
        const inpaintingResponse = await axiosInstance.post('/diffusion/inpainting', {
          prompt: inpaintingPrompt.current.value,
          mask: masksResponse.data,
          postprocess_mode: postprocessMode,
          is_applying_blur: isApplyingBlur,
          using_canny_control_image: usingCannyControl,
          num_inference_steps: numInferenceSteps,
          guidance_scale: guidanceScale,
          controlnet_conditioning_scale: controlnetScale,
          num_samples: numSamples
        }, { responseType: 'blob' })

        // Create a URL from the Blob
        const imageInpaintingUrl = URL.createObjectURL(inpaintingResponse.data)
        
        // Update the image state with the new URL
        setImageInpainting(imageInpaintingUrl)
      } catch (error) {
        console.error('Error in text segmentation:', error)
        setApiResponse('Error in text segmentation')
      }
    }
  }

  const handleFileChange = async (event) => {
    const file = event.target.files[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => setImageSegment(e.target.result)
      reader.readAsDataURL(file)

      const formData = new FormData()
      formData.append('image', file)
      try {
        const response_dino = await axiosInstance.post('/groundingdino/set_image', formData)
        console.log(response_dino.data.message)
        const response_sam = await axiosInstance.post('/sam2/add-image', formData)
        console.log(response_sam.data.message)
        const response_diffusion = await axiosInstance.post('/diffusion/set_image', formData)
        console.log(response_diffusion.data.message)
        setApiResponse(response_sam.data.message)
      } catch (error) {
        console.error('Error uploading image:', error)
        setApiResponse('Error uploading image')
      }
    }
  }

  const handleApiCall = async () => {
    try {
      const response = await axiosInstance.get('/sam2/')
      console.log(response.data)
      setApiResponse(response.data.message)
    } catch (error) {
      console.error('Error calling API:', error)
      setApiResponse('Error calling API')
    }
  }

  const updateSliderBackground = (e) => {
    const target = e.target;
    const min = target.min;
    const max = target.max;
    const val = target.value;
    const percentage = (val - min) * 100 / (max - min);
    target.style.backgroundSize = `${percentage}% 100%`;
  }

  return (
    <div className="max-w-6xl mx-auto p-8 font-sans">
      <header className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-800">Image Segmentation with SAM2</h1>
      </header>
      <main className="flex flex-row gap-8">
        {/* Left Section - 50% width */}
        <div className="w-5/10">
          <div className="mb-8">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
              id="fileInput"
            />
            <label
              htmlFor="fileInput"
              className="block w-full px-6 py-3 bg-blue-500 text-white font-bold rounded-lg cursor-pointer transition duration-300 hover:bg-blue-600 mb-4 text-center"
            >
              Choose an image
            </label>
            <button
              onClick={handleApiCall}
              className="block w-full px-6 py-3 bg-green-500 text-white font-bold rounded-lg cursor-pointer transition duration-300 hover:bg-green-600 text-center"
            >
              Test API
            </button>
          </div>
          <div className="mb-4 p-4 border border-gray-300 rounded-lg">
            <h3 className="font-semibold text-gray-700 mb-3">Settings</h3>
            <div className="space-y-4">
              <div className="flex items-center space-x-4">
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="singleTargetMode"
                    checked={singleTargetMode}
                    onChange={() => setSingleTargetMode(!singleTargetMode)}
                    className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
                  />
                  <label htmlFor="singleTargetMode" className="ml-2 text-sm text-gray-700">
                    Single Target Mode
                  </label>
                </div>
                <div className="flex items-center">
                  <input 
                    type="checkbox" 
                    id="postprocessMode" 
                    checked={postprocessMode}
                    onChange={() => setPostprocessMode(!postprocessMode)}
                    className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
                  />
                  <label htmlFor="postprocessMode" className="ml-2 text-sm text-gray-700">
                    Postprocess Mode
                  </label>
                </div>
              </div>

              <div className="flex items-center space-x-4">
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="isApplyingBlur"
                    checked={isApplyingBlur}
                    onChange={() => setIsApplyingBlur(!isApplyingBlur)}
                    className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
                  />
                  <label htmlFor="isApplyingBlur" className="ml-2 text-sm text-gray-700">
                    Apply Mask Blur
                  </label>
                </div>
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="usingCannyControl"
                    checked={usingCannyControl}
                    onChange={() => setUsingCannyControl(!usingCannyControl)}
                    className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
                  />
                  <label htmlFor="usingCannyControl" className="ml-2 text-sm text-gray-700">
                    Use Canny Control
                  </label>
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <div className="flex justify-between mb-2">
                    <label htmlFor="numInferenceSteps" className="block text-sm font-medium text-gray-700">
                      Inference Steps
                    </label>
                    <span className="text-sm text-blue-600 font-medium">{numInferenceSteps}</span>
                  </div>
                  <input
                    type="range"
                    id="numInferenceSteps"
                    value={numInferenceSteps}
                    onChange={(e) => {
                      setNumInferenceSteps(Number(e.target.value));
                      updateSliderBackground(e);
                    }}
                    onInput={updateSliderBackground}
                    min="1"
                    max="50"
                    step="1"
                    className="w-full"
                    style={{ backgroundSize: `${(numInferenceSteps - 1) * 100 / 49}% 100%` }}
                  />
                </div>

                <div>
                  <div className="flex justify-between mb-2">
                    <label htmlFor="guidanceScale" className="block text-sm font-medium text-gray-700">
                      Guidance Scale
                    </label>
                    <span className="text-sm text-blue-600 font-medium">{guidanceScale}</span>
                  </div>
                  <input
                    type="range"
                    id="guidanceScale"
                    value={guidanceScale}
                    onChange={(e) => {
                      setGuidanceScale(Number(e.target.value));
                      updateSliderBackground(e);
                    }}
                    onInput={updateSliderBackground}
                    min="1"
                    max="15"
                    step="0.5"
                    className="w-full"
                    style={{ backgroundSize: `${(guidanceScale - 1) * 100 / 14}% 100%` }}
                  />
                </div>

                <div>
                  <div className="flex justify-between mb-2">
                    <label htmlFor="controlnetScale" className="block text-sm font-medium text-gray-700">
                      ControlNet Scale
                    </label>
                    <span className="text-sm text-blue-600 font-medium">{controlnetScale}</span>
                  </div>
                  <input
                    type="range"
                    id="controlnetScale"
                    value={controlnetScale}
                    onChange={(e) => {
                      setControlnetScale(Number(e.target.value));
                      updateSliderBackground(e);
                    }}
                    onInput={updateSliderBackground}
                    min="0"
                    max="1"
                    step="0.02"
                    className="w-full"
                    style={{ backgroundSize: `${controlnetScale * 100}% 100%` }}
                  />
                </div>

                <div>
                  <div className="flex justify-between mb-2">
                    <label htmlFor="numSamples" className="block text-sm font-medium text-gray-700">
                      Number of Samples
                    </label>
                    <span className="text-sm text-blue-600 font-medium">{numSamples}</span>
                  </div>
                  <input
                    type="range"
                    id="numSamples"
                    value={numSamples}
                    onChange={(e) => {
                      setNumSamples(Number(e.target.value));
                      updateSliderBackground(e);
                    }}
                    onInput={updateSliderBackground}
                    min="0"
                    max="5"
                    step="1"
                    className="w-full"
                    style={{ backgroundSize: `${numSamples * 100 / 5}% 100%` }}
                  />
                </div>
              </div>
            </div>
          </div>
          <textarea
            className="w-full h-24 p-4 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter text prompt here..."
            onFocus={() => {
              segmentPrompt.current.style.borderColor = "gray"
              segmentPrompt.current.style.borderWidth = "1px"
            }}
            ref={segmentPrompt}
          />
          <textarea
            className="w-full h-24 p-4 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter inpainting prompt here..."
            onFocus={() => {
              inpaintingPrompt.current.style.borderColor = "gray"
              inpaintingPrompt.current.style.borderWidth = "1px"
            }}
            ref={inpaintingPrompt}
          />
          <button
            onClick={handleTextSegment}
            disabled={imageMaskRef.current === null}
            className="mt-3 block w-full px-6 py-3 bg-blue-950 text-white font-bold rounded-lg cursor-pointer transition duration-300 hover:bg-blue-900 text-center disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Segment
          </button>

          {apiResponse && (
            <div className="mt-4 p-4 bg-gray-100 rounded-lg">
              <p>API Response: {apiResponse}</p>
            </div>
          )}
        </div>

        {/* Right Section - 50% width */}
        <div className="w-5/10">
          <div className="relative w-full h-fit bg-gray-100 rounded-lg flex items-center justify-center mb-4 max-w-[500px]">
            {imageSegment ? (
              <div className="relative w-full h-full">
                <img
                  src={imageSegment}
                  alt="Uploaded image"
                  onClick={handleImageClick}
                  ref={imageMaskRef}
                  className="w-full h-full object-contain rounded-lg shadow-lg cursor-crosshair"
                />
                {coordinates && (
                  <span className="absolute bottom-0 left-0 bg-black bg-opacity-70 text-white px-3 py-2 rounded-br-lg text-sm">
                    Coordinates: ({coordinates.x}, {coordinates.y})
                  </span>
                )}
              </div>
            ) : (
              <div className="text-gray-400 text-lg">
                Please upload an image
              </div>
            )}
          </div>

          <div className="relative w-full h-fit bg-gray-100 rounded-lg flex items-center justify-center">
            {imageInpainting ? (
              <div className="relative w-full h-full">
                <img
                  src={imageInpainting}
                  alt="Inpainted image"
                  ref={imageInpaintingRef}  
                  className="w-full h-full object-contain rounded-lg shadow-lg"
                />
              </div>
            ) : (
              <div className="text-gray-400 text-lg">
                Inpainting result will appear here
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
