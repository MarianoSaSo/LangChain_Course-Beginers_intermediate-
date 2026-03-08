-- SCRIPT DE INICIALIZACIÓN: Base de Datos 'world'
-- Ejecuta este script en MySQL Workbench para tener datos de prueba.

CREATE DATABASE IF NOT EXISTS world;
USE world;

-- 1. Crear la tabla country
CREATE TABLE IF NOT EXISTS country (
    Code CHAR(3) PRIMARY KEY,
    Name VARCHAR(52) NOT NULL,
    Continent ENUM('Asia','Europe','North America','Africa','Oceania','Antarctica','South America') NOT NULL,
    Region VARCHAR(26) NOT NULL,
    Population INT NOT NULL,
    LifeExpectancy FLOAT
);

-- 2. Insertar datos de prueba
INSERT IGNORE INTO country (Code, Name, Continent, Region, Population, LifeExpectancy) VALUES
('CHN', 'China', 'Asia', 'Eastern Asia', 1277558000, 71.4),
('IND', 'India', 'Asia', 'Southern and Central Asia', 1013662000, 62.5),
('JPN', 'Japan', 'Asia', 'Eastern Asia', 126714000, 80.7),
('ESP', 'Spain', 'Europe', 'Southern Europe', 39441700, 78.8),
('FRA', 'France', 'Europe', 'Western Europe', 59225700, 78.8),
('DEU', 'Germany', 'Europe', 'Western Europe', 82164700, 77.4),
('USA', 'United States', 'North America', 'North America', 278357000, 77.1),
('BRA', 'Brazil', 'South America', 'South America', 170115000, 62.9),
('RUS', 'Russian Federation', 'Europe', 'Eastern Europe', 146934000, 67.2);

-- 3. Crear una tabla simple de ciudades para que el agente vea que hay varias tablas
CREATE TABLE IF NOT EXISTS city (
    ID INT PRIMARY KEY AUTO_INCREMENT,
    Name VARCHAR(35) NOT NULL,
    CountryCode CHAR(3),
    Population INT,
    FOREIGN KEY (CountryCode) REFERENCES country(Code)
);

INSERT IGNORE INTO city (Name, CountryCode, Population) VALUES
('Madrid', 'ESP', 2877000),
('Barcelona', 'ESP', 1503000),
('Paris', 'FRA', 2125246),
('Berlin', 'DEU', 3386667),
('Tokyo', 'JPN', 7980230),
('Beijing', 'CHN', 7480000);
