# Dynamic Solutions System

This system allows developers to edit problem solutions without touching HTML files. All problem content is now stored in JSON files and dynamically loaded by the HTML pages.

## How It Works

### 1. **Solution Files** (`solutions/` directory)
- Each problem has a corresponding JSON file (e.g., `two-sum.json`)
- Contains all problem information: description, solutions, code, complexity analysis, etc.
- Developers can edit these files directly to update solutions

### 2. **Dynamic HTML Template**
- All problem HTML files now use a single dynamic template
- Automatically loads content from JSON solution files
- No hardcoded content - everything is dynamic

### 3. **Automatic Updates**
- Changes to JSON files are immediately reflected in the HTML
- No need to regenerate HTML files
- Just commit changes to JSON files and they appear on the website

## File Structure

```
TopAlgorithms/
├── problems/                    # HTML files (now dynamic templates)
│   ├── two-sum.html           # Dynamic template
│   ├── 3sum.html             # Dynamic template
│   └── ...
├── solutions/                  # JSON solution files
│   ├── two-sum.json          # Problem data and solutions
│   ├── 3sum.json            # Problem data and solutions
│   └── ...
├── extract_solutions.py       # Script to extract solutions from HTML
├── convert_to_dynamic.py      # Script to convert HTML to dynamic
├── dynamic_problem_template.html  # Dynamic template
└── index.html                 # Main index page
```

## JSON Solution File Format

Each solution file follows this structure:

```json
{
  "problem": {
    "title": "Problem Title",
    "leetcode_id": 1,
    "category": "String/Array/Matrix",
    "description": "Problem description...",
    "assumptions": ["Assumption 1", "Assumption 2"],
    "example": {
      "input": "Input example",
      "output": "Output example",
      "explanation": "Explanation..."
    }
  },
  "solutions": [
    {
      "title": "Solution Title",
      "description": "Solution description...",
      "code": "public int[] solution() { ... }",
      "language": "java",
      "complexity": {
        "time": "O(n)",
        "space": "O(1)"
      },
      "example_walkthrough": {
        "steps": ["Step 1", "Step 2"]
      }
    }
  ],
  "variations": [
    {
      "title": "Variation Title",
      "description": "Variation description...",
      "code": "Code for variation..."
    }
  ],
  "navigation": {
    "previous": "prev-problem.html",
    "next": "next-problem.html"
  }
}
```

## How to Use

### For Developers (Updating Solutions)

1. **Edit JSON files directly** in the `solutions/` directory
2. **No need to touch HTML files**
3. **Changes are automatically reflected** when the page is loaded
4. **Commit JSON changes** to update the website

### Example: Adding a New Solution

1. Open `solutions/two-sum.json`
2. Add a new solution to the `solutions` array:

```json
{
  "title": "Solution 4 – New Approach",
  "description": "Description of the new solution...",
  "code": "public int[] newSolution(int[] nums, int target) { ... }",
  "language": "java",
  "complexity": {
    "time": "O(n log n)",
    "space": "O(n)"
  }
}
```

3. Save the file
4. The new solution automatically appears on the website

### Example: Updating Code

1. Open any JSON solution file
2. Modify the `code` field
3. Save the file
4. The updated code appears on the website immediately

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Extract Solutions from Existing HTML

```bash
python extract_solutions.py
```

This will:
- Parse all existing HTML files
- Extract problem information, solutions, and code
- Create JSON files in the `solutions/` directory

### 3. Convert HTML Files to Dynamic

```bash
python convert_to_dynamic.py
```

This will:
- Convert all HTML files to use the dynamic template
- Create backups of original files (`.html.backup`)
- Make all pages load content from JSON files

### 4. Test the System

1. Open any problem HTML file in a browser
2. Verify that content loads from JSON files
3. Edit a JSON file and refresh the page to see changes

## Benefits

### ✅ **Developer Experience**
- Edit solutions without touching HTML
- Simple JSON format for content
- Version control friendly
- No HTML knowledge required

### ✅ **Maintenance**
- Centralized content management
- Easy to update multiple problems
- Consistent formatting across all pages
- No duplicate code

### ✅ **Performance**
- Lazy loading of content
- Smaller HTML files
- Better caching potential
- Faster page loads

### ✅ **Scalability**
- Easy to add new problems
- Simple to modify existing content
- Automated content extraction
- Template-based system

## Troubleshooting

### Problem: Content not loading
- Check that JSON files exist in `solutions/` directory
- Verify JSON syntax is valid
- Check browser console for errors
- Ensure file paths are correct

### Problem: Styling issues
- Verify CSS is properly loaded
- Check that dynamic template is used
- Ensure HTML structure is maintained

### Problem: Navigation not working
- Check navigation links in JSON files
- Verify HTML filenames match
- Ensure proper file extensions

## Migration from Old System

### Before (Hardcoded HTML)
- Content embedded in HTML files
- Manual updates required
- HTML knowledge needed
- Difficult to maintain

### After (Dynamic JSON)
- Content in separate JSON files
- Automatic updates
- No HTML knowledge needed
- Easy to maintain

## Contributing

1. **Fork the repository**
2. **Edit JSON solution files** to update solutions
3. **Test your changes** by opening the HTML pages
4. **Commit and push** your JSON changes
5. **No HTML changes needed**

## Support

If you encounter issues:
1. Check the browser console for errors
2. Verify JSON file syntax
3. Ensure all required files are present
4. Check file permissions and paths

---

**Note**: This system maintains the exact same visual appearance and functionality while making content management much easier for developers. 