# Frontend Changes - Dark/Light Theme Toggle

## Overview
Added a theme toggle button that allows users to switch between dark and light themes.

## Files Modified

### 1. `frontend/style.css`

#### CSS Variables
- Added light theme CSS variables under `[data-theme="light"]` selector
- Light theme includes:
  - Light background color (`#f8fafc`)
  - White surface color (`#ffffff`)
  - Dark text for contrast (`#1e293b`)
  - Adjusted border and shadow colors

#### Theme Toggle Button Styles
- Fixed position in top-right corner
- Circular button (44px diameter)
- Sun icon (yellow) displayed in light theme
- Moon icon (gray) displayed in dark theme
- Smooth rotation and opacity transitions on theme switch
- Hover, focus, and active states
- Responsive sizing for mobile devices

#### Transition Animations
- Added smooth 0.3s ease transitions for theme-aware elements:
  - Container, sidebar, chat container
  - Message content, stat items, suggested items
  - Input fields and buttons

### 2. `frontend/index.html`

#### Theme Toggle Button
- Added button element with `id="themeToggle"` in the body
- Contains two SVG icons:
  - Sun icon (for switching to light theme)
  - Moon icon (for switching to dark theme)
- Includes accessibility attributes:
  - `aria-label="Toggle theme"`
  - `title="Toggle dark/light theme"`

#### Version Updates
- Updated CSS and JS file versions from `v=9` to `v=10`

### 3. `frontend/script.js`

#### Theme Management Functions
- `getPreferredTheme()`: Checks localStorage first, then system preference
- `setTheme(theme)`: Sets `data-theme` attribute on `<html>` element and stores in localStorage
- `toggleTheme()`: Switches between 'dark' and 'light' themes

#### Event Listeners
- Click event on theme toggle button
- Keyboard accessibility (Enter and Space keys)

#### Theme Persistence
- Theme preference stored in localStorage under key `theme-preference`
- Respects system color scheme preference on first visit

## Implementation Details

### Theme Switching Mechanism
1. Theme is controlled via `data-theme` attribute on the `<html>` element
2. CSS variables change based on the attribute value
3. All colors automatically update through CSS variable references

### Accessibility Features
- Button is keyboard navigable
- ARIA label for screen readers
- Focus ring for visual indication
- Title attribute for tooltip

### User Experience
- Smooth 0.3s transitions between themes
- Theme preference persisted across sessions
- Respects system preference on first visit
- Icons animate with rotation effect during toggle