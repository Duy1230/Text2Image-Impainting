import { useState, useRef, useEffect } from 'react'
import axiosInstance from '../axiosInstance'

export default function Component() {
  const [image, setImage] = useState(null)
  const [coordinates, setCoordinates] = useState(null)
  const [apiResponse, setApiResponse] = useState(null)
  const imageRef = useRef(null)

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

  const handleFileChange = async (event) => {
    const file = event.target.files[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => setImage(e.target.result)
      reader.readAsDataURL(file)

      const formData = new FormData()
      formData.append('image', file)
      try {
        const response = await axiosInstance.post('/sam2/add-image', formData)
        console.log(response.data.message)
        setApiResponse(response.data.message)
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
    <div className="max-w-4xl mx-auto p-8 font-sans">
      <header className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-800">Image Segmentation with SAM2</h1>
      </header>
      <main className="flex flex-col items-center">
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
            className="inline-block px-6 py-3 bg-blue-500 text-white font-bold rounded-lg cursor-pointer transition duration-300 hover:bg-blue-600 mr-4"
          >
            Choose an image
          </label>
          <button
            onClick={handleApiCall}
            className="inline-block px-6 py-3 bg-green-500 text-white font-bold rounded-lg cursor-pointer transition duration-300 hover:bg-green-600"
          >
            Call API
          </button>
        </div>
        {apiResponse && (
          <div className="mb-4 p-4 bg-gray-100 rounded-lg">
            <p>API Response: {apiResponse}</p>
          </div>
        )}
        {image && (
          <div className="relative max-w-full">
            <img
              src={image}
              alt="Uploaded image"
              onClick={handleImageClick}
              ref={imageRef}
              className="max-w-full h-auto rounded-lg shadow-lg cursor-crosshair"
            />
            {coordinates && (
              <span className="absolute bottom-0 left-0 bg-black bg-opacity-70 text-white px-3 py-2 rounded-br-lg text-sm">
                Coordinates: ({coordinates.x}, {coordinates.y})
              </span>
            )}
          </div>
        )}
      </main>
    </div>
  )
}
