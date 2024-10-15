import { useState, useRef } from 'react'

export default function Component() {
  const [image, setImage] = useState(null)
  const [coordinates, setCoordinates] = useState(null)
  const imageRef = useRef(null)

  const handleImageClick = (event) => {
    if (imageRef.current) {
      const rect = imageRef.current.getBoundingClientRect()
      const x = Math.round(event.clientX - rect.left)
      const y = Math.round(event.clientY - rect.top)
      setCoordinates({ x, y })
    }
  }

  const handleFileChange = (event) => {
    const file = event.target.files[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => setImage(e.target.result)
      reader.readAsDataURL(file)
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
            className="inline-block px-6 py-3 bg-blue-500 text-white font-bold rounded-lg cursor-pointer transition duration-300 hover:bg-blue-600"
          >
            Choose an image
          </label>
        </div>
        {image && (
          <div className="relative max-w-full">
            <img
              src={image}
              alt="Uploaded image"
              onClick={handleImageClick}
              ref={imageRef}
              className="max-w-full h-auto rounded-lg shadow-lg cursor-crosshair"
            />
            
          </div>
        )}
      </main>
    </div>
  )
}