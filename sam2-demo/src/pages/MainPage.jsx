import { useState, useRef, useEffect } from 'react'
import axiosInstance from '../axiosInstance'

export default function Component() {
  const [image, setImage] = useState(null)
  const [coordinates, setCoordinates] = useState(null)
  const [apiResponse, setApiResponse] = useState(null)
  const imageRef = useRef(null)
  const textPrompt = useRef(null)

  useEffect(() => {
    // Cleanup the object URL when the component unmounts or image changes
    return () => {
      if (image) {
        URL.revokeObjectURL(image)
      }
    }
  }, [image])

  const handleImageClick = async (event) => {
    if (imageRef.current) {
      const rect = imageRef.current.getBoundingClientRect()
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
        setImage(imageUrl)
      } catch (error) {
        console.error('Error segmenting image:', error)
        setApiResponse('Error segmenting image')
      }
    }
  }

  const handleTextSegment = async () => {
    if (textPrompt.current.value === "") {
      setApiResponse("Please enter a text prompt")
      //change border to red
      textPrompt.current.style.borderColor = "red"
      textPrompt.current.style.borderWidth = "2px"
      textPrompt.current.placeholder = "Please enter a text prompt"
    } else {
      try {
        // First get the boxes from GroundingDINO
        const dinoResponse = await axiosInstance.post('/groundingdino/predict', { 
          prompt: textPrompt.current.value 
        })
        
        // Then use the boxes for SAM2 segmentation
        const response = await axiosInstance.post('/sam2/segment_with_text', {
          boxes: dinoResponse.data.boxes
        }, { responseType: 'blob' })
        
        // Create a URL from the Blob
        const imageUrl = URL.createObjectURL(response.data)
        
        // Update the image state with the new URL
        setImage(imageUrl)
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
      reader.onload = (e) => setImage(e.target.result)
      reader.readAsDataURL(file)

      const formData = new FormData()
      formData.append('image', file)
      try {
        const response_dino = await axiosInstance.post('/groundingdino/set_image', formData)
        console.log(response_dino.data.message)
        const response_sam = await axiosInstance.post('/sam2/add-image', formData)
        console.log(response_sam.data.message)
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

  return (
    <div className="max-w-6xl mx-auto p-8 font-sans">
      <header className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-800">Image Segmentation with SAM2</h1>
      </header>
      <main className="flex flex-row gap-8">
        {/* Left Section - 30% width */}
        <div className="w-3/10">
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

          <textarea
            className="w-full h-48 p-4 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter text prompt here..."
            onFocus={() => {
              textPrompt.current.style.borderColor = "gray"
              textPrompt.current.style.borderWidth = "1px"
            }}
            ref={textPrompt}
          />
          <button
            onClick={handleTextSegment}
            className="mt-3 block w-full px-6 py-3 bg-blue-950 text-white font-bold rounded-lg cursor-pointer transition duration-300 hover:bg-blue-900 text-center"
          >
            Segment
          </button>

          {apiResponse && (
            <div className="mt-4 p-4 bg-gray-100 rounded-lg">
              <p>API Response: {apiResponse}</p>
            </div>
          )}
        </div>

        {/* Right Section - 70% width */}
        <div className="w-7/10">
          <div className="relative w-full h-fit bg-gray-100 rounded-lg flex items-center justify-center">
            {image ? (
              <div className="relative w-full h-full">
                <img
                  src={image}
                  alt="Uploaded image"
                  onClick={handleImageClick}
                  ref={imageRef}
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
        </div>
      </main>
    </div>
  )
}
