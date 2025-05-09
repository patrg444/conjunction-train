/**
 * Simplified Heatmap Implementation
 * For visualization of eye tracking data
 */

function createHeatmap(config) {
    const canvas = config.canvas;
    const ctx = canvas.getContext('2d');
    const radius = config.radius || 25;
    const maxOpacity = config.maxOpacity || 0.6;
    const minOpacity = config.minOpacity || 0;
    const blur = config.blur || 0.85;
    
    let points = [];
    let max = 1; // Max point value
    
    // Add data points to the heatmap
    function addData(newPoints) {
        // If single point, convert to array
        if (!Array.isArray(newPoints)) {
            newPoints = [newPoints];
        }
        
        points = points.concat(newPoints);
        
        // Update max value if needed
        for (let i = 0; i < newPoints.length; i++) {
            const point = newPoints[i];
            max = Math.max(max, point.value || 1);
        }
        
        // Draw the updated points
        drawHeatmap();
    }
    
    // Clear all points
    function clear() {
        points = [];
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    
    // Set data (replaces existing data)
    function setData(newPoints) {
        clear();
        addData(newPoints);
    }
    
    // Draw heatmap based on current points
    function drawHeatmap() {
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Nothing to draw
        if (points.length === 0) return;
        
        // Create initial invisible canvas for generating gradients
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = canvas.width;
        tempCanvas.height = canvas.height;
        const tempCtx = tempCanvas.getContext('2d');
        
        // Draw gradient for each point
        for (let i = 0; i < points.length; i++) {
            const point = points[i];
            const x = point.x;
            const y = point.y;
            const value = point.value || 1;
            
            // Create radial gradient
            const gradient = tempCtx.createRadialGradient(
                x, y, 0, 
                x, y, radius
            );
            
            // Calculate alpha based on value
            const alpha = (value / max) * maxOpacity;
            
            // Set gradient stops
            gradient.addColorStop(0, `rgba(255, 0, 0, ${alpha})`);
            gradient.addColorStop(0.5, `rgba(255, 255, 0, ${alpha * 0.6})`);
            gradient.addColorStop(1, `rgba(0, 255, 0, 0)`);
            
            // Draw point with gradient
            tempCtx.fillStyle = gradient;
            tempCtx.beginPath();
            tempCtx.arc(x, y, radius, 0, 2 * Math.PI);
            tempCtx.fill();
        }
        
        // Apply blur if specified
        if (blur > 0) {
            ctx.filter = `blur(${blur * 10}px)`;
        }
        
        // Draw temp canvas to main canvas
        ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);
        
        // Reset filter
        ctx.filter = 'none';
    }
    
    // Return public API
    return {
        addData: addData,
        setData: setData,
        clear: clear,
        getCanvas: () => canvas,
        getData: () => points
    };
}
