// NOTE: ALL VALUES IN THIS FILE ARE IN METERS
#pragma once

#include "vec.hpp"
#include <vector>
#include <stdexcept>

#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "std_msgs/msg/color_rgba.hpp"

enum class CellValue {
    FREE = 0,
    OCCUPIED,
    PLAUSIBLE_PARKING,
    CONFIRMED_PARKING,
    UNKNOWN,
    COUNT
};

class Cell {
    private:
        vec2f m_cellPosition;
        vec2f m_cellSize;

    public:
        CellValue value;

    public:
        Cell() = default;
        Cell(vec2f position, vec2f size, CellValue value = CellValue::UNKNOWN) : m_cellPosition(position), m_cellSize(size), value(value) { }

        const vec2f& getPosition() const { return m_cellPosition; }
};

class Grid {
    private:
        vec2f m_gridSize;
        vec2f m_cellSize;
        vec2f m_origin;

        std::vector<Cell> m_grid {};

        uint32_t CELL_NUM;
        vec2u CELL_COUNT;

    public:
        Grid(vec2f gridSize_cm, vec2f cellSize_cm, vec2f origin) {
            m_gridSize = gridSize_cm;
            m_cellSize = cellSize_cm;
            m_origin = origin;

            vec2f cellCountF = m_gridSize / m_cellSize;

            CELL_COUNT = { static_cast<uint32_t>(std::ceil(cellCountF.x)), static_cast<uint32_t>(std::ceil(cellCountF.y)) };
            CELL_NUM = CELL_COUNT.x * CELL_COUNT.y;

            m_grid.reserve(CELL_NUM);
            for (uint32_t i = 0; i < CELL_COUNT.x; ++i) {
                for (uint32_t j = 0; j < CELL_COUNT.y; ++j) {
                    m_grid.emplace_back(m_origin + m_cellSize * vec2i(i, j), m_cellSize, CellValue::FREE);
                }
            }
        }

        size_t getFlattenIndex(const vec2u& index) const {
            return index.x + index.y * CELL_COUNT.x;
        }

        Cell& get(const vec2u& index) {
            return m_grid[getFlattenIndex(index)]; 
        }

        // Returns the index of the cell at 'position' relative to the center
        vec2i getRelativeIndex(const vec2f& position) {
            return vec2i {
                static_cast<int32_t>(std::floor(position.x / m_cellSize.x)),
                static_cast<int32_t>(std::floor(position.y / m_cellSize.y))
            };
        }

        Cell& getFromCenter(const vec2i& index) {
            vec2i half { static_cast<int32_t>(CELL_COUNT.x / 2),
                         static_cast<int32_t>(CELL_COUNT.y / 2) };
            vec2i shifted = half + index;
        
            // Bounds check — a center-relative index can easily fall outside the
            // grid (e.g. an obstacle detected beyond your grid's radius), and
            // get() has no bounds checking of its own (raw vector indexing).
            if (shifted.x < 0 || shifted.y < 0 ||
                static_cast<uint32_t>(shifted.x) >= CELL_COUNT.x ||
                static_cast<uint32_t>(shifted.y) >= CELL_COUNT.y) {
                throw std::out_of_range("getFromCenter: index outside grid bounds");
            }
        
            vec2u realIndex { static_cast<uint32_t>(shifted.x),
                              static_cast<uint32_t>(shifted.y) };
            return get(realIndex);
        }


        template<typename Func>
        void forEach(Func&& func) {
            for (uint32_t i = 0; i < CELL_COUNT.x; ++i) {
                for (uint32_t j = 0; j < CELL_COUNT.y; ++j) {
                    vec2u index(i, j);
                    func(get(index), index);
                }
            }
        }

        visualization_msgs::msg::Marker toMarker(const std::string& frame_id, rclcpp::Time stamp) const {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = frame_id;
            marker.header.stamp = stamp;
            marker.ns = "occupancy_grid";
            marker.id = 0;
            marker.type = visualization_msgs::msg::Marker::CUBE_LIST;
            marker.action = visualization_msgs::msg::Marker::ADD;
        
            marker.scale.x = m_cellSize.x;
            marker.scale.y = m_cellSize.y;
            marker.scale.z = 0.02;
        
            marker.pose.orientation.w = 1.0;

            marker.color.r = 1.0f;
            marker.color.g = 1.0f;
            marker.color.b = 1.0f;
            marker.color.a = 1.0f;
        
            for (uint32_t i = 0; i < CELL_COUNT.x; ++i) {
                for (uint32_t j = 0; j < CELL_COUNT.y; ++j) {
                    const Cell& cell = m_grid[getFlattenIndex({j, i})];
        
                    geometry_msgs::msg::Point p;
                    p.x = (m_origin.x + m_cellSize.x * i);
                    p.y = (m_origin.y + m_cellSize.y * j);
                    p.z = 0.0;
                    marker.points.push_back(p);
        
                    std_msgs::msg::ColorRGBA color;
                    color.a = 0.7f;
                    switch (cell.value) {
                        case CellValue::FREE:              color.g = 1.0f; break;                     // green
                        case CellValue::OCCUPIED:          color.r = 1.0f; break;                      // red
                        case CellValue::PLAUSIBLE_PARKING: color.r = 1.0f; color.g = 1.0f; break;       // yellow
                        case CellValue::CONFIRMED_PARKING: color.b = 1.0f; break;                       // blue
                        case CellValue::UNKNOWN:
                        default:                           color.r = color.g = color.b = 0.4f; break;   // grey
                    }
                    marker.colors.push_back(color);
                }
            }
            return marker;
        }

};
